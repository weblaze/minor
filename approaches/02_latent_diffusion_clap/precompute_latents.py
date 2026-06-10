"""Encode every paired image to its VAE latent once, so diffusion training
never loads the VAE or CLAP — minimal VRAM on the GTX 1650 and faster epochs.

Run after make_pairs.py. Re-run if you change vae.kind or retrain the VAE.

Usage:
  python precompute_latents.py [--batch-size 32]
"""
import argparse
import os
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abstraction.data.datasets import ClapImageDataset
from abstraction.models.vae_codec import build_codec
from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    dataset = ClapImageDataset(
        pairs_file=config["paths"]["pairs_file"],
        clap_dir=config["paths"]["clap_features"],
        image_dir=config["paths"]["abstract_art"],
        size=config["image_size"],
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    codec = build_codec(config, device)

    out_dir = config["paths"]["latents_dir"]
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    for batch in loader:
        latents = codec.encode(batch["image"].to(device)).cpu()
        for latent, audio_emb in zip(latents, batch["audio_emb"]):
            torch.save({"latent": latent, "audio_emb": audio_emb},
                       os.path.join(out_dir, f"{count:06d}.pt"))
            count += 1
        print(f"  encoded {count}/{len(dataset)}", end="\r")
    print(f"\nWrote {count} latent records to {out_dir}")


if __name__ == "__main__":
    main()
