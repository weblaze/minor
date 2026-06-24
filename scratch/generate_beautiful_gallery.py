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
    audio_files = [
        Path("datasets/fma_small/000/000002.mp3"),
        Path("datasets/fma_small/000/000005.mp3"),
        Path("datasets/fma_small/000/000010.mp3"),
        Path("datasets/fma_small/000/000140.mp3"),
        Path("datasets/fma_small/000/000141.mp3")
    ]
    
    config = load_config(Path("approaches/01_conditioned_autoencoder/config.yaml"))
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    n_samples = config["gallery"]["samples_per_song"]
    
    model = ConditionedImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, config["paths"]["conditioned_checkpoint"], device=device)
    model.eval()
    
    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)
    
    torch.manual_seed(config["system"]["seed"])
    raw_noise = torch.randn(n_samples, config["image_vae"]["latent_channels"], 16, 16, device=device)
    
    # Apply spatial smoothing (Gaussian filter)
    kernel_size = 5
    sigma = 1.5
    x = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
    x_grid, y_grid = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(config["image_vae"]["latent_channels"], 1, 1, 1).to(device)
    
    latents = F.conv2d(raw_noise, kernel, padding=kernel_size//2, groups=config["image_vae"]["latent_channels"])
    latents = (latents / latents.std()) * 2.0  # Scale standard deviation to 2.0
    
    rows = []
    for audio in audio_files:
        print(f"embedding {audio.name}...")
        cond = clap.embed_file(audio).repeat(n_samples, 1)
        with torch.no_grad():
            images = model.decode(latents, cond=cond)
        rows.append((images + 1) / 2)
        
    grid = vutils.make_grid(torch.cat(rows).clamp(0, 1).cpu(), nrow=n_samples)
    out_path = Path("outputs/01_conditioned_autoencoder/genre_gallery_smoothed.png")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)
    print(f"Gallery saved to {out_path}")
    
    # Print statistics of the grid
    arr = grid.numpy()
    print(f"Mean RGB: {arr.mean(axis=(1, 2))}")
    print(f"Std RGB:  {arr.std(axis=(1, 2))}")

if __name__ == "__main__":
    main()
