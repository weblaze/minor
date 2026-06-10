"""t-SNE views of the two latent spaces: ImageVAE latents and CLAP embeddings.

The spaces have different dimensionalities (spatial 8x16x16 vs 512) so each
gets its own projection. Look for structure: CLAP should cluster by genre/mood;
a healthy VAE latent space should be spread, not collapsed to a point.

Usage:
  python visualization/visualize_latent_space.py [--num-samples 200]
"""
import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Subset

from abstraction.data.datasets import ImageDataset
from abstraction.models.image_vae import ImageVAE
from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config

OUTPUT_DIR = Path(__file__).parent / "latent_space_plots"


def tsne_2d(latents):
    if np.any(~np.isfinite(latents)):
        latents = latents[np.isfinite(latents).all(axis=1)]
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(latents) - 1))
    return tsne.fit_transform(latents)


def image_latents(config, device, num_samples, batch_size=8):
    ckpt = config["paths"]["image_vae_checkpoint"]
    if not os.path.exists(ckpt):
        print(f"No ImageVAE checkpoint at {ckpt} — skipping image latent plot")
        return None
    model = ImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, ckpt, device=device)
    model.eval()

    dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"])
    subset = Subset(dataset, range(min(num_samples, len(dataset))))
    loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

    latents = []
    with torch.no_grad():
        for batch in loader:
            mu, logvar, _ = model.encode(batch.to(device))
            z = model.reparameterize(mu, torch.clamp(logvar, -10, 10))
            latents.append(z.view(z.size(0), -1).cpu().numpy())
    return np.concatenate(latents)


def clap_embeddings(config, num_samples):
    clap_dir = config["paths"]["clap_features"]
    if not os.path.isdir(clap_dir):
        print(f"No CLAP features at {clap_dir} — skipping CLAP plot")
        return None
    files = sorted(f for f in os.listdir(clap_dir) if f.endswith(".npy"))[:num_samples]
    if not files:
        return None
    return np.stack([np.load(os.path.join(clap_dir, f)) for f in files])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num-samples", type=int, default=200)
    args = parser.parse_args()

    config = load_config(Path(__file__).parents[1] / "approaches" / "01_conditioned_autoencoder" / "config.yaml")
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panels = []
    print("Extracting image VAE latents...")
    img = image_latents(config, device, args.num_samples)
    if img is not None:
        panels.append(("ImageVAE latent space", img, "tab:blue"))
    print("Loading CLAP embeddings...")
    clap = clap_embeddings(config, args.num_samples)
    if clap is not None:
        panels.append(("CLAP audio embedding space", clap, "tab:red"))

    if not panels:
        print("Nothing to plot — need a trained ImageVAE and/or extracted CLAP features.")
        return

    fig, axes = plt.subplots(1, len(panels), figsize=(8 * len(panels), 7))
    if len(panels) == 1:
        axes = [axes]
    for ax, (title, latents, color) in zip(axes, panels):
        print(f"t-SNE: {title} ({latents.shape[0]} samples, dim {latents.shape[1]})")
        coords = tsne_2d(latents)
        ax.scatter(coords[:, 0], coords[:, 1], c=color, alpha=0.6, s=14)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)

    out_path = OUTPUT_DIR / "latent_space_distribution.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    print(f"Saved to {out_path}")


if __name__ == "__main__":
    main()
