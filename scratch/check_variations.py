import torch
import torchvision.utils as vutils
from pathlib import Path
import os
import torch.nn.functional as F

from abstraction.audio.clap import ClapEncoder
from abstraction.models.image_vae import ConditionedImageVAE
from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config

def main():
    config = load_config(Path("approaches/01_conditioned_autoencoder/config.yaml"))
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    n_samples = 4
    
    model = ConditionedImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, config["paths"]["conditioned_checkpoint"], device=device)
    model.eval()
    
    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)
    audio_file = "datasets/fma_small/000/000002.mp3"
    cond = clap.embed_file(audio_file).repeat(n_samples, 1)
    
    torch.manual_seed(42)
    raw_noise = torch.randn(n_samples, config["image_vae"]["latent_channels"], 16, 16, device=device)
    
    # Method 1: Scale raw noise by 2.0
    latents_scaled = raw_noise * 2.0
    
    # Method 2: Scale by 3.0
    latents_scaled_3 = raw_noise * 3.0

    # Method 3: Spatial smoothing (blurring) to introduce correlation + scale
    # Create a simple Gaussian blur kernel
    kernel_size = 5
    sigma = 1.5
    x = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
    x_grid, y_grid = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(config["image_vae"]["latent_channels"], 1, 1, 1).to(device)
    
    # Smooth raw noise and then scale to have std of ~2.0
    latents_smoothed = F.conv2d(raw_noise, kernel, padding=kernel_size//2, groups=config["image_vae"]["latent_channels"])
    latents_smoothed = (latents_smoothed / latents_smoothed.std()) * 2.0

    with torch.no_grad():
        recon_raw = model.decode(raw_noise, cond=cond)
        recon_scaled = model.decode(latents_scaled, cond=cond)
        recon_scaled_3 = model.decode(latents_scaled_3, cond=cond)
        recon_smoothed = model.decode(latents_smoothed, cond=cond)
        
    os.makedirs("evaluation/results", exist_ok=True)
    vutils.save_image((recon_raw + 1)/2, "evaluation/results/test_raw.png")
    vutils.save_image((recon_scaled + 1)/2, "evaluation/results/test_scaled_2.png")
    vutils.save_image((recon_scaled_3 + 1)/2, "evaluation/results/test_scaled_3.png")
    vutils.save_image((recon_smoothed + 1)/2, "evaluation/results/test_smoothed.png")
    
    # Print statistics
    def print_stats(name, tensor):
        arr = (tensor + 1)/2
        print(f"{name}:")
        print(f"  Mean: {arr.mean(dim=(0, 2, 3)).cpu().numpy()}")
        print(f"  Std:  {arr.std(dim=(0, 2, 3)).cpu().numpy()}")
        
    print_stats("Raw", recon_raw)
    print_stats("Scaled 2.0", recon_scaled)
    print_stats("Scaled 3.0", recon_scaled_3)
    print_stats("Smoothed 2.0", recon_smoothed)

if __name__ == "__main__":
    main()
