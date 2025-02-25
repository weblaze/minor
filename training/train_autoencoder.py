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
import numpy as np

# Enhanced CUDA memory management
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Set memory allocation configuration
if torch.cuda.is_available():
    # More aggressive memory settings
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128,garbage_collection_threshold:0.8,roundup_power2_divisions:16'
    # Empty cache and collect garbage
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
    """Enhanced diversity loss calculation with better regularization."""
    # Normalize features for stable calculations
    features = F.instance_norm(features)
    
    # Spatial diversity - encourage variation across spatial dimensions
    spatial_std = torch.std(features, dim=[2, 3])  # [B, C]
    spatial_diversity = -torch.mean(torch.log(spatial_std + 1e-6))
    
    # Channel diversity - encourage variation across channels
    channel_std = torch.std(features, dim=1)  # [B, H, W]
    channel_diversity = -torch.mean(torch.log(channel_std + 1e-6))
    
    # Batch diversity - encourage variation across batch
    batch_features = features.mean(dim=[2, 3])  # [B, C]
    batch_features = F.normalize(batch_features, dim=1)
    similarity = torch.mm(batch_features, batch_features.t())
    batch_diversity = torch.mean(similarity) - torch.mean(torch.diagonal(similarity))
    
    # Color balance - encourage balanced color distribution
    color_mean = features.mean(dim=[0, 2, 3])  # [C]
    color_balance = torch.mean((color_mean - 0.5).pow(2))
    
    # Combine with balanced weights
    total_div = (
        0.3 * spatial_diversity +
        0.3 * channel_diversity +
        0.3 * batch_diversity +
        0.1 * color_balance
    )
    
    return torch.clamp(total_div, -1.0, 1.0)

def kl_loss(mu, logvar):
    """Enhanced KL divergence loss with better stability and regularization."""
    # Clamp values for stability
    mu = torch.clamp(mu, -10, 10)
    logvar = torch.clamp(logvar, -10, 10)
    
    # Standard KL divergence
    kld = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Add regularization terms
    mu_reg = 0.01 * torch.mean(torch.abs(mu))  # L1 on mean
    var_reg = 0.01 * torch.mean(torch.abs(logvar.exp() - 1))  # Encourage unit variance
    
    # Combine losses
    total_kl = torch.mean(kld) + mu_reg + var_reg
    return torch.clamp(total_kl, 0.0, 10.0)

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

    # Load dataset with optimized settings
    print("Loading dataset...")
    dataloader = get_dataloader(
        audio_features_path, 
        image_folder, 
        batch_size=4,  # Reduced batch size for better memory management
        num_workers=2,
        pin_memory=True,
        prefetch_factor=2,  # Reduced prefetch factor
        normalize_features=True,
        augment_images=True,
        augment_audio=True
    )

    # Initialize model with memory optimizations
    print("Initializing model...")
    model = AudioToImageAutoencoder(image_size=128).to(device)
    
    # Enable gradient checkpointing for all supported layers
    def enable_grad_checkpointing(module):
        if hasattr(module, 'gradient_checkpointing'):
            module.gradient_checkpointing = True
        if hasattr(module, 'enable_gradient_checkpointing'):
            module.enable_gradient_checkpointing()
    
    model.apply(enable_grad_checkpointing)
    
    # Use DataParallel only if multiple GPUs are available and enough memory
    if torch.cuda.device_count() > 1 and torch.cuda.get_device_properties(0).total_memory > 8e9:  # 8GB
        model = torch.nn.DataParallel(model)
    
    # Initialize gradient scaler with more aggressive settings
    scaler = GradScaler(
        init_scale=65536,    # 2^16
        growth_interval=100,  # More frequent scaling updates
        growth_factor=2.0,    # More aggressive scaling
        backoff_factor=0.5,   # Less aggressive backoff
        enabled=True
    )
    
    # Improved optimizer settings
    optimizer = optim.AdamW(
        model.parameters(),
        lr=0.0001,  # Lower initial learning rate
        betas=(0.5, 0.999),
        weight_decay=0.02,  # Increased weight decay
        eps=1e-8
    )
    
    # Enhanced training configuration
    num_epochs = 100  # Double the epochs
    warmup_epochs = 10  # Longer warmup
    save_every = 5
    analyze_features_every = 5
    
    # Improved learning rate schedule
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.0003,
        epochs=num_epochs,
        steps_per_epoch=len(dataloader),
        pct_start=0.2,  # Faster warmup
        div_factor=25.0,  # Larger lr range
        final_div_factor=1000.0,
        anneal_strategy='cos'
    )

    # Enhanced KL annealing with cyclic schedule
    kl_weight_min = 0.0
    kl_weight_max = 0.1
    kl_cycle_epochs = 10  # Cycle KL weight every 10 epochs
    
    # Initialize VGG for perceptual loss
    vgg = VGGFeatureExtractor(device)
    
    # Initialize latent space monitoring
    latent_stats = {
        'mu_mean': [], 'mu_std': [],
        'logvar_mean': [], 'logvar_std': [],
        'z_activity': []  # Track which latent dimensions are active
    }
    
    print(f"Starting training for {num_epochs} epochs")
    print(f"Total iterations per epoch: {len(dataloader)}")
    print(f"Total training steps: {num_epochs * len(dataloader)}")
    
    for epoch in range(num_epochs):
        model.train()
        total_recon_loss = total_kl_loss = total_perceptual_loss = total_div_loss = 0
        epoch_mu = []
        epoch_logvar = []
        
        # Calculate cyclic KL weight
        progress = (epoch % kl_cycle_epochs) / kl_cycle_epochs
        if epoch < warmup_epochs:
            # Linear warmup during warmup epochs
            kl_weight = kl_weight_max * (epoch / warmup_epochs)
        else:
            # Cyclic schedule after warmup
            kl_weight = kl_weight_min + 0.5 * (kl_weight_max - kl_weight_min) * \
                       (1 + np.cos(progress * np.pi))
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for batch_idx, (audio, image) in enumerate(pbar):
            # Move data to GPU with non_blocking and explicit stream
            current_stream = torch.cuda.current_stream()
            with torch.cuda.stream(current_stream):
                audio = audio.to(device, non_blocking=True)
                image = image.to(device, non_blocking=True)
            
            # Ensure data transfer is complete
            current_stream.synchronize()
            
            optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            try:
                with torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                    # Generate images with memory optimization
                    generated, mu, logvar = model(audio)
                    
                    # Store latent variables for monitoring
                    epoch_mu.append(mu.detach().cpu())
                    epoch_logvar.append(logvar.detach().cpu())
                    
                    # Calculate losses with memory-efficient operations
                    recon_loss = F.mse_loss(generated, image, reduction='mean')
                    perceptual = perceptual_loss(vgg, generated, image)
                    
                    # Only calculate diversity loss occasionally to save memory
                    if batch_idx % 4 == 0:
                        div_loss = diversity_loss(generated)
                    else:
                        div_loss = torch.tensor(0.0, device=device)
                    
                    kl_div = kl_weight * kl_loss(mu, logvar)
                    
                    # Dynamic loss weighting based on training progress
                    recon_weight = 1.0
                    perceptual_weight = min(0.3, 0.1 + 0.2 * (epoch / num_epochs))
                    div_weight = min(0.2, 0.05 + 0.15 * (epoch / num_epochs))
                    
                    # Combined loss
                    loss = (
                        recon_weight * recon_loss +
                        perceptual_weight * perceptual +
                        div_weight * div_loss +
                        kl_div
                    )
                    
                    # Collect data for feature analysis
                    if (epoch + 1) % analyze_features_every == 0:
                        analyzer.analyze_image_quality(generated)
                    
                    # Check for NaN and reset if found
                    if torch.isnan(loss):
                        print(f"\nNaN detected! Skipping batch {batch_idx}")
                        print(f"recon_loss: {recon_loss.item():.4f}, perceptual: {perceptual.item():.4f}, "
                              f"div_loss: {div_loss.item():.4f}, kl_div: {kl_div.item():.4f}")
                        optimizer.zero_grad(set_to_none=True)
                        continue
                    
                    # Clear unnecessary tensors
                    del mu, logvar
                
                # Memory-efficient backward pass
                scaler.scale(loss).backward()
                
                # Unscale before clip to handle potential inf/nan
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Update metrics every 10 batches
                if batch_idx % 10 == 0:
                    total_recon_loss += recon_loss.item()
                    total_perceptual_loss += perceptual.item()
                    total_kl_loss += kl_div.item()
                    total_div_loss += div_loss.item()
                    
                    # Update progress bar
                    pbar.set_postfix({
                        'loss': f"{loss.item():.4f}",
                        'recon': f"{recon_loss.item():.4f}",
                        'percep': f"{perceptual.item():.4f}",
                        'lr': f"{scheduler.get_last_lr()[0]:.6f}"
                    })
                
                # Analyze image quality
                if batch_idx % 50 == 0:
                    analyzer.analyze_image_quality(generated)
                
                # Aggressive cleanup after each step
                del loss, recon_loss, perceptual, div_loss, kl_div, generated
                torch.cuda.empty_cache()
                
                # Force garbage collection periodically
                if batch_idx % 10 == 0:
                    gc.collect()
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"\nCUDA OOM in batch {batch_idx}. Clearing cache and reducing batch...")
                    torch.cuda.empty_cache()
                    gc.collect()
                    continue
                else:
                    print(f"Error in batch {batch_idx}: {str(e)}")
                    continue
            
            # Aggressive variable cleanup
            torch.cuda.empty_cache()
        
        # Calculate epoch metrics
        num_batches = len(dataloader) // 10
        avg_recon = total_recon_loss / num_batches
        avg_perceptual = total_perceptual_loss / num_batches
        avg_kl = total_kl_loss / num_batches
        avg_div = total_div_loss / num_batches
        
        # Monitor latent space
        epoch_mu = torch.cat(epoch_mu, dim=0)
        epoch_logvar = torch.cat(epoch_logvar, dim=0)
        
        latent_stats['mu_mean'].append(epoch_mu.mean().item())
        latent_stats['mu_std'].append(epoch_mu.std().item())
        latent_stats['logvar_mean'].append(epoch_logvar.mean().item())
        latent_stats['logvar_std'].append(epoch_logvar.std().item())
        
        # Calculate latent dimension activity
        z_activity = (torch.abs(epoch_mu) > 0.1).float().mean(0)
        latent_stats['z_activity'].append(z_activity.numpy())
        
        # Log enhanced metrics
        analyzer.log_epoch_metrics(
            avg_recon, avg_perceptual, avg_kl, avg_div,
            kl_weight=kl_weight,
            active_dims=z_activity.mean().item(),
            mu_stats={'mean': latent_stats['mu_mean'][-1], 
                     'std': latent_stats['mu_std'][-1]},
            logvar_stats={'mean': latent_stats['logvar_mean'][-1],
                         'std': latent_stats['logvar_std'][-1]}
        )
        
        print(f"\nEpoch {epoch+1} Summary:")
        print(f"Recon: {avg_recon:.4f}, Perceptual: {avg_perceptual:.4f}")
        print(f"KL: {avg_kl:.4f} (weight: {kl_weight:.4f}), Div: {avg_div:.4f}")
        print(f"Active latent dims: {(z_activity > 0.1).sum().item()}/{len(z_activity)}")
        
        # Save checkpoints with enhanced metadata
        if (epoch + 1) % save_every == 0:
            checkpoint_path = f"models/autoencoder/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'loss': avg_recon,
                'latent_stats': latent_stats,
                'kl_weight': kl_weight
            }, checkpoint_path)
            
            # Generate and save visualization plots
            analyzer.plot_training_progress(latent_stats)
            
            # Clear memory after checkpoint
            torch.cuda.empty_cache()
            gc.collect()
        
        # Memory cleanup between epochs
        torch.cuda.empty_cache()
        gc.collect()
    
    print("Training completed!")
    
    # Save final analysis with latent space statistics
    analyzer.save_metrics(latent_stats)
    analyzer.generate_report(latent_stats)

if __name__ == "__main__":
    train() 