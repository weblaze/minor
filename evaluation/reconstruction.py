"""Side-by-side VAE reconstruction check: 5 originals on top, reconstructions below.

Usage:
  python evaluation/reconstruction.py [--ckpt path/to/image_vae.pth]
"""
import argparse
import os
from pathlib import Path

import torch
import torchvision.utils as vutils
from torch.utils.data import DataLoader

from abstraction.data.datasets import ImageDataset
from abstraction.models.image_vae import ImageVAE
from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ckpt", type=str, default=None,
                        help="ImageVAE checkpoint (default: approach 01 pretrain checkpoint)")
    parser.add_argument("--out", type=str, default="evaluation/results/vae_reconstruction_test.png")
    args = parser.parse_args()

    config = load_config(Path(__file__).parents[1] / "approaches" / "01_conditioned_autoencoder" / "config.yaml")
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    ckpt = args.ckpt or config["paths"]["image_vae_checkpoint"]
    if not os.path.exists(ckpt):
        print(f"Checkpoint not found at {ckpt} — train approach 01 stage pretrain first.")
        return

    model = ImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, ckpt, device=device)
    model.eval()
    print(f"Loaded checkpoint from {ckpt}")

    dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    batch = next(iter(DataLoader(dataset, batch_size=5, shuffle=True)))
    images = batch.to(device)

    with torch.no_grad():
        recon_images, _, _ = model(images)

    combined = torch.cat([images, recon_images], dim=0)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(combined, out_path, nrow=5, normalize=True)
    print(f"Saved to {out_path} — top row originals, bottom row reconstructions.")


if __name__ == "__main__":
    main()
