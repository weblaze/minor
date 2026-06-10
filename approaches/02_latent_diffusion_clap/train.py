"""Train the conditional UNet denoiser in VAE latent space.

Conditioning: CLAP audio embeddings (or BYOL-A via cond.source for approach 03).
Classifier-free guidance is built in from day one: with probability cfg_dropout
the conditioning is replaced with zeros, so the same model supports both
audio-conditioned generation and null-conditioned dreaming.

Prerequisites: make_pairs.py and precompute_latents.py have been run.

Usage:
  python train.py [--epochs N] [--max-steps N] [--no-wandb]
"""
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from abstraction.data.datasets import LatentDataset
from abstraction.models.scheduler import DDPMScheduler
from abstraction.models.unet import ConditionalUNet
from abstraction.utils.checkpoints import save_checkpoint
from abstraction.utils.config import load_config
from abstraction.utils.wandb_utils import init_wandb

APPROACH = "02_latent_diffusion_clap"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=None, help="override config num_epochs")
    parser.add_argument("--max-steps", type=int, default=None, help="stop after N optimizer steps (smoke test)")
    parser.add_argument("--no-wandb", action="store_true")
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    diff = config["latent_diffusion"]
    epochs = args.epochs or diff["num_epochs"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["system"]["seed"])
    print(f"Using device: {device}")

    dataset = LatentDataset(config["paths"]["latents_dir"])
    loader = DataLoader(dataset, batch_size=diff["batch_size"], shuffle=True,
                        drop_last=True, num_workers=0)
    print(f"Loaded {len(dataset)} precomputed latent records")

    sample = dataset[0]
    latent_channels = sample["latent"].shape[0]
    unet = ConditionalUNet(
        in_channels=latent_channels,
        out_channels=latent_channels,
        time_emb_dim=diff["time_emb_dim"],
        condition_dim=config["cond"]["dim"],
    ).to(device)

    optimizer = torch.optim.Adam(unet.parameters(), lr=float(diff["learning_rate"]))
    scheduler = DDPMScheduler(num_train_timesteps=diff["num_train_timesteps"])
    scheduler.set_device(device)
    cfg_dropout = diff["cfg_dropout"]
    ckpt_path = config["paths"]["unet_checkpoint"]

    run = None if args.no_wandb else init_wandb(config, APPROACH, "train")

    unet.train()
    step = 0
    for epoch in range(epochs):
        epoch_loss = 0.0
        for batch in loader:
            latents = batch["latent"].to(device)
            cond = batch["audio_emb"].to(device)

            # classifier-free guidance: randomly null out conditioning
            drop = torch.rand(cond.shape[0], device=device) < cfg_dropout
            cond = torch.where(drop[:, None], torch.zeros_like(cond), cond)

            noise = torch.randn_like(latents)
            timesteps = torch.randint(0, scheduler.num_train_timesteps,
                                      (latents.shape[0],), device=device).long()
            noisy = scheduler.add_noise(latents, noise, timesteps)

            optimizer.zero_grad()
            noise_pred = unet(noisy, timesteps, cond)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()

            epoch_loss += loss.item()
            step += 1
            if args.max_steps and step >= args.max_steps:
                save_checkpoint(unet, ckpt_path)
                print(f"[smoke] stopped after {step} steps, checkpoint saved to {ckpt_path}")
                return

        avg_loss = epoch_loss / len(loader)
        print(f"Epoch {epoch + 1}/{epochs} | avg loss {avg_loss:.4f}", flush=True)
        if run:
            run.log({"epoch": epoch + 1, "loss": avg_loss})

        if (epoch + 1) % 10 == 0 or (epoch + 1) == epochs:
            save_checkpoint(unet, ckpt_path)
            print(f"--- checkpoint saved at epoch {epoch + 1} ---", flush=True)

    save_checkpoint(unet, ckpt_path)
    print(f"Saved trained UNet to {ckpt_path}")


if __name__ == "__main__":
    main()
