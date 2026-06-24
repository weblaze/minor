import torch
from torch.utils.data import DataLoader
from abstraction.data.datasets import ImageDataset
from abstraction.models.image_vae import ImageVAE
from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config
from pathlib import Path

def main():
    config = load_config(Path("approaches/01_conditioned_autoencoder/config.yaml"))
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    
    model = ImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, config["paths"]["image_vae_checkpoint"], device=device)
    model.eval()
    
    dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    images = next(iter(loader)).to(device)
    
    with torch.no_grad():
        mu, logvar, _ = model.encode(images)
        std = torch.exp(0.5 * logvar)
        
    print(f"Mu shape: {mu.shape}")
    print(f"Mu mean: {mu.mean().item():.4f}")
    print(f"Mu std:  {mu.std().item():.4f}")
    print(f"Mu min:  {mu.min().item():.4f}")
    print(f"Mu max:  {mu.max().item():.4f}")
    print("-" * 30)
    print(f"Std mean: {std.mean().item():.4f}")
    print(f"Std std:  {std.std().item():.4f}")
    print(f"Std min:  {std.min().item():.4f}")
    print(f"Std max:  {std.max().item():.4f}")

if __name__ == "__main__":
    main()
