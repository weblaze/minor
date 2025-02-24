import torch
import torch.nn as nn
import torch.optim as optim
from training.data_loader_audio_img import get_dataloader
from models.autoencoder.model import AudioToImageAutoencoder
from training.analyze_training import create_analyzer
import os
from pathlib import Path
import torch.nn.functional as F

def gram_matrix(features):
    batch_size, channels, height, width = features.size()
    features = features.view(batch_size, channels, height * width)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (channels * height * width)
    return gram

def style_loss(generated, target):
    # Extract features at different scales
    scales = [(8, 8), (16, 16), (32, 32)]
    total_loss = 0
    
    for scale in scales:
        g_resized = F.interpolate(generated, size=scale)
        t_resized = F.interpolate(target, size=scale)
        
        # Calculate Gram matrices
        g_gram = gram_matrix(g_resized)
        t_gram = gram_matrix(t_resized)
        
        # Add to total loss
        total_loss += F.mse_loss(g_gram, t_gram)
    
    return total_loss

def diversity_loss(features):
    # Encourage diversity in generated images
    gram = gram_matrix(features)
    return -torch.mean(torch.std(gram, dim=0))

def kl_loss(mu, logvar):
    """Calculate KL divergence loss."""
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def perceptual_loss(generated, target):
    """Calculate perceptual loss using multiple scales."""
    scales = [(8, 8), (16, 16), (32, 32)]
    total_loss = 0
    
    for scale in scales:
        g_scaled = F.interpolate(generated, size=scale)
        t_scaled = F.interpolate(target, size=scale)
        total_loss += F.mse_loss(g_scaled, t_scaled)
    
    return total_loss

def train():
    # Set paths
    audio_features_path = "data/normalized_features.npy"
    image_folder = "datasets/abstract_art"
    save_model_path = "models/autoencoder/audio_to_image_autoencoder.pth"

    # Create save directory
    os.makedirs(os.path.dirname(save_model_path), exist_ok=True)

    # Initialize analyzer
    analyzer = create_analyzer()

    # Load dataset
    print("Loading dataset...")
    dataloader = get_dataloader(audio_features_path, image_folder, batch_size=8)

    # Check for CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Initialize model
    print("Initializing model...")
    model = AudioToImageAutoencoder().to(device)
    reconstruction_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0002, betas=(0.5, 0.999))

    # Training Loop
    print("Starting training...")
    num_epochs = 50
    for epoch in range(num_epochs):
        total_recon_loss = 0
        total_style_loss = 0
        total_div_loss = 0
        total_kl_loss = 0
        total_perceptual_loss = 0
        
        for batch_idx, (audio, image) in enumerate(dataloader):
            audio, image = audio.to(device), image.to(device)
            optimizer.zero_grad()
            
            # Generate images
            generated, mu, logvar = model(audio)
            
            # Calculate losses
            recon_loss = reconstruction_criterion(generated, image)
            s_loss = style_loss(generated, image)
            div_loss = diversity_loss(generated)
            kl_div = kl_loss(mu, logvar)
            percep_loss = perceptual_loss(generated, image)
            
            # Combine losses with adjusted weights
            loss = (recon_loss + 
                   0.1 * s_loss + 
                   0.05 * div_loss + 
                   0.01 * kl_div +
                   0.1 * percep_loss)
            
            loss.backward()
            optimizer.step()
            
            # Log batch metrics
            analyzer.log_batch_metrics(
                recon_loss.item(),
                s_loss.item(),
                div_loss.item(),
                loss.item()
            )
            
            # Analyze image quality
            analyzer.analyze_image_quality(generated)
            
            total_recon_loss += recon_loss.item()
            total_style_loss += s_loss.item()
            total_div_loss += div_loss.item()
            total_kl_loss += kl_div.item()
            total_perceptual_loss += percep_loss.item()
            
            if batch_idx % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx}/{len(dataloader)}], "
                      f"Loss: {loss.item():.4f}, Style: {s_loss.item():.4f}, "
                      f"Div: {div_loss.item():.4f}, KL: {kl_div.item():.4f}")

        # Calculate and log epoch metrics
        avg_recon = total_recon_loss / len(dataloader)
        avg_style = total_style_loss / len(dataloader)
        avg_div = total_div_loss / len(dataloader)
        avg_kl = total_kl_loss / len(dataloader)
        avg_percep = total_perceptual_loss / len(dataloader)
        
        analyzer.log_epoch_metrics(avg_recon, avg_style, avg_div)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], "
              f"Recon: {avg_recon:.4f}, Style: {avg_style:.4f}, "
              f"Div: {avg_div:.4f}, KL: {avg_kl:.4f}, "
              f"Percep: {avg_percep:.4f}")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = f"models/autoencoder/checkpoint_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_recon,
            }, checkpoint_path)
            print(f"Checkpoint saved to {checkpoint_path}")
            
            # Generate and save analysis plots
            analyzer.plot_training_progress()
            analyzer.save_metrics()

    # Save the final model
    print(f"Saving model to {save_model_path}")
    torch.save(model.state_dict(), save_model_path)
    
    # Generate final analysis
    analyzer.plot_training_progress()
    analyzer.save_metrics()
    analyzer.generate_report()
    print("Training completed! Analysis results saved in data/training_analysis/")

if __name__ == "__main__":
    train() 