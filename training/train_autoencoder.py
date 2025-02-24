import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from training.data_loader_audio_img import get_dataloader
from models.autoencoder.model import AudioToImageAutoencoder
from training.analyze_training import create_analyzer
import os
from pathlib import Path
import torch.nn.functional as F
from torchvision.models import vgg16, VGG16_Weights
from torchvision import transforms
from torch.amp import autocast, GradScaler
import gc
from tqdm import tqdm

# Set CUDA configurations for better GPU performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

# Set memory allocation configuration
if torch.cuda.is_available():
    # Enable expandable segments to avoid fragmentation
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512,expandable_segments:True'
    # Empty cache
    torch.cuda.empty_cache()
    gc.collect()

class VGGFeatureExtractor(nn.Module):
    def __init__(self, device):
        super(VGGFeatureExtractor, self).__init__()
        # Use VGG16 for feature extraction
        vgg = vgg16(weights=VGG16_Weights.IMAGENET1K_V1)
        
        # Extract features from multiple layers
        self.slice1 = nn.Sequential(*list(vgg.features)[:4]).eval()  # After first conv block
        self.slice2 = nn.Sequential(*list(vgg.features)[4:9]).eval()  # After second conv block
        self.slice3 = nn.Sequential(*list(vgg.features)[9:16]).eval()  # After third conv block
        
        # Freeze parameters
        for param in self.parameters():
            param.requires_grad = False
        
        # Register normalization values
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        
        self.to(device)
    
    def normalize(self, x):
        """Normalize input images similar to ImageNet normalization."""
        return (x - self.mean) / self.std
    
    @torch.amp.autocast('cuda')
    def forward(self, x):
        # Ensure input is in range [0, 1]
        x = torch.clamp(x, 0, 1)
        # Normalize input
        x = self.normalize(x)
        # Get features from different depths
        h1 = self.slice1(x)
        h2 = self.slice2(h1)
        h3 = self.slice3(h2)
        return [h1, h2, h3]

def gram_matrix(features):
    """Calculate Gram Matrix with numerical stability."""
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, -1)
    features_t = features.transpose(1, 2)
    
    # Add small epsilon for numerical stability
    gram = torch.bmm(features, features_t)
    # Normalize by feature map size
    return gram / (channels * height * width + 1e-8)

def perceptual_loss(vgg, generated, target):
    """Calculate perceptual loss using VGG features with proper scaling and stability."""
    # Get features
    gen_features = vgg(generated)
    target_features = vgg(target)
    
    content_loss = 0
    style_loss = 0
    
    # Weights for different feature levels (deeper layers have higher weights for style)
    content_weights = [1.0, 0.75, 0.25]  # Focus on lower-level features for content
    style_weights = [0.25, 0.75, 1.0]    # Focus on higher-level features for style
    
    for gen_feat, target_feat, c_weight, s_weight in zip(
        gen_features, target_features, content_weights, style_weights):
        
        if torch.any(torch.isnan(gen_feat)) or torch.any(torch.isnan(target_feat)):
            continue
        
        # Content loss - compare features directly
        feat_loss = F.mse_loss(gen_feat, target_feat)
        content_loss += c_weight * feat_loss
        
        # Style loss - compare Gram matrices
        gen_gram = gram_matrix(gen_feat)
        target_gram = gram_matrix(target_feat)
        gram_loss = F.mse_loss(gen_gram, target_gram)
        style_loss += s_weight * gram_loss
    
    # Balance content and style losses
    total_loss = content_loss + 0.5 * style_loss
    return total_loss

def diversity_loss(features):
    """Simplified and stable diversity loss calculation."""
    # Clamp features to prevent extreme values
    features = torch.clamp(features, -10, 10)
    
    # Quick spatial diversity with stability
    spatial_std = torch.clamp(features.std(dim=[2, 3]), min=1e-6)  # [B, C]
    spatial_diversity = -torch.mean(torch.log1p(spatial_std))
    
    # Quick batch diversity with stability
    mean_features = features.mean(dim=[2, 3])  # [B, C]
    mean_features = F.normalize(mean_features, dim=1)  # Normalize for stable distances
    batch_diversity = -torch.mean(torch.pdist(mean_features))
    
    # Simplified color balance with stability
    color_mean = features.mean(dim=[0, 2, 3])  # [C]
    color_balance = torch.mean((color_mean - 0.5).pow(2))
    
    # Combine with scaling
    total_div = (
        0.1 * spatial_diversity +
        0.1 * batch_diversity +
        0.01 * color_balance
    )
    
    return torch.clamp(total_div, -1.0, 1.0)

def kl_loss(mu, logvar):
    """Calculate KL divergence loss with stability measures."""
    # Clamp values for stability
    mu = torch.clamp(mu, -10, 10)
    logvar = torch.clamp(logvar, -10, 10)
    
    # Calculate KL divergence with numerical stability
    kl_div = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return torch.clamp(kl_div, 0.0, 10.0)

def train():
    # Set paths
    audio_features_path = "data/normalized_features.npy"
    image_folder = "datasets/abstract_art"
    save_model_path = "models/autoencoder/audio_to_image_autoencoder.pth"

    # Create save directory
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # Initialize analyzer
    analyzer = create_analyzer()

    # Enhanced CUDA settings
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()
        # Optimize CUDA performance
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU")

    # Load dataset with enhanced settings
    print("Loading dataset...")
    dataloader = get_dataloader(
        audio_features_path, 
        image_folder, 
        batch_size=8,
        num_workers=2,
        pin_memory=True,
        normalize_features=True,
        augment_images=True,
        augment_audio=True  # Enable audio augmentation
    )

    # Initialize model
    print("Initializing model...")
    model = AudioToImageAutoencoder().to(device)
    
    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, 'enable_gradient_checkpointing'):
        model.enable_gradient_checkpointing()
    
    # Use DataParallel only if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model)
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Memory-optimized training parameters
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0005,
        betas=(0.9, 0.999),
        weight_decay=0.01,
        eps=1e-8
    )
    
    # Training configuration
    num_epochs = 20
    warmup_epochs = 2
    save_every = 4
    analyze_features_every = 5  # Analyze feature importance every 5 epochs
    
    # Learning rate schedule with gentler changes
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.0005,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.2,
        anneal_strategy='cos',
        div_factor=5.0,
        final_div_factor=50.0
    )

    # KL annealing parameters with gentler ramp-up
    kl_weight = 0.0
    kl_weight_max = 0.1
    kl_warmup_steps = warmup_epochs * len(dataloader)
    kl_step = 0
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Total iterations per epoch: {len(dataloader)}")
    print(f"Total training steps: {num_epochs * len(dataloader)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_recon_loss = total_kl_loss = total_div_loss = 0
        
        # Clear memory at start of epoch
        torch.cuda.empty_cache()
        gc.collect()
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        # Collect batches for feature analysis
        if (epoch + 1) % analyze_features_every == 0:
            epoch_audio_features = []
            epoch_generated_images = []
        
        for batch_idx, (audio, image) in enumerate(pbar):
            # Non-blocking GPU transfer
            audio = audio.to(device, non_blocking=True)
            image = image.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)
            
            try:
                device_type = 'cuda' if torch.cuda.is_available() else 'cpu'
                with torch.amp.autocast(device_type):
                    # Generate images
                    generated, mu, logvar = model(audio)
                    
                    # Calculate reconstruction loss with stability
                    recon_loss = F.mse_loss(
                        torch.clamp(generated, 0, 1),
                        torch.clamp(image, 0, 1)
                    )
                    recon_loss = torch.clamp(recon_loss, 0.0, 10.0)
                    
                    # Calculate diversity loss with reduced frequency
                    if batch_idx % 4 == 0:
                        div_loss = 0.1 * diversity_loss(generated)
                    else:
                        div_loss = torch.tensor(0.0, device=device)
                    
                    # KL loss with gentler annealing
                    kl_weight = min(kl_weight_max, 0.5 * (1 - torch.cos(torch.tensor(kl_step / kl_warmup_steps) * torch.pi)))
                    kl_div = kl_weight * kl_loss(mu, logvar)
                    kl_step += 1
                    
                    # Combined loss with stability check
                    loss = recon_loss + div_loss + kl_div
                    
                    # Collect data for feature analysis
                    if (epoch + 1) % analyze_features_every == 0:
                        epoch_audio_features.append(audio.detach().cpu())
                        epoch_generated_images.append(generated.detach().cpu())
                    
                    # Check for NaN and reset if found
                    if torch.isnan(loss):
                        print(f"\nNaN detected! Skipping batch {batch_idx}")
                        print(f"recon_loss: {recon_loss.item():.4f}, div_loss: {div_loss.item():.4f}, kl_div: {kl_div.item():.4f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Clear intermediate tensors
                    del mu, logvar
                
                # Optimized backward pass with gradient clipping
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Update metrics every 10 batches
                if batch_idx % 10 == 0:
                    total_recon_loss += recon_loss.item()
                    total_kl_loss += kl_div.item()
                    total_div_loss += div_loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'div': f"{div_loss.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                
                # Analyze image quality
                if batch_idx % 50 == 0:
                    analyzer.analyze_image_quality(generated)
                
                # Clear memory every 25 batches
                if batch_idx % 25 == 0:
                    torch.cuda.empty_cache()
                
            except RuntimeError as e:
                print(f"Error in batch {batch_idx}: {str(e)}")
                continue
            
            # Aggressive variable cleanup
            del loss, recon_loss, div_loss, kl_div, generated, audio, image
            torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        avg_recon = total_recon_loss / (len(dataloader) // 10)
        avg_kl = total_kl_loss / (len(dataloader) // 10)
        avg_div = total_div_loss / (len(dataloader) // 10)
        
        # Log epoch metrics
        analyzer.log_epoch_metrics(avg_recon, avg_kl, avg_div)
        
        # Analyze feature importance
        if (epoch + 1) % analyze_features_every == 0:
            print("\nAnalyzing feature importance...")
            epoch_audio_features = torch.cat(epoch_audio_features, dim=0)
            epoch_generated_images = torch.cat(epoch_generated_images, dim=0)
            analyzer.analyze_feature_importance(epoch_audio_features, epoch_generated_images)
            print("Feature importance analysis completed")
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Recon: {avg_recon:.4f}, KL: {avg_kl:.4f}, Div: {avg_div:.4f}")
        
        # Save checkpoints
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"models/autoencoder/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_recon,
            }, checkpoint_path)
            
            # Generate and save visualization plots
            analyzer.plot_training_progress()
            
            # Clear memory after checkpoint
            torch.cuda.empty_cache()
            gc.collect()
    
    print("Training completed!")
    
    # Save final analysis
    analyzer.save_metrics()
    analyzer.generate_report()

if __name__ == "__main__":
    train() 