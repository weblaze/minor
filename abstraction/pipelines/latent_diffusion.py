import os

import numpy as np
import torch

from abstraction.audio.clap import CLAP_DIM, ClapEncoder
from abstraction.models.scheduler import DDPMScheduler
from abstraction.models.unet import ConditionalUNet
from abstraction.models.vae_codec import build_codec
from abstraction.utils.checkpoints import load_checkpoint


def load_models(config):
    """Load the CLAP encoder, diffusion UNet, VAE codec, and scheduler from config."""
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    diff = config["latent_diffusion"]

    print(f"Loading CLAP on {device}...")
    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)

    print("Loading VAE codec...")
    codec = build_codec(config, device)

    print("Loading diffusion UNet...")
    unet = ConditionalUNet(
        in_channels=codec.latent_channels,
        out_channels=codec.latent_channels,
        time_emb_dim=diff["time_emb_dim"],
        condition_dim=config.get("cond", {}).get("dim", CLAP_DIM),
    ).to(device)
    unet_path = config["paths"]["unet_checkpoint"]
    if os.path.exists(unet_path):
        load_checkpoint(unet, unet_path, device=device)
        print(f"[pipeline] loaded UNet from {unet_path}")
    else:
        print(f"[pipeline] WARNING: no UNet checkpoint at {unet_path} — using untrained weights")
    unet.eval()

    scheduler = DDPMScheduler(num_train_timesteps=diff["num_train_timesteps"])
    scheduler.set_device(device)

    return clap, unet, codec, scheduler


@torch.no_grad()
def generate_diffusion(clap, unet, codec, scheduler, audio_path=None,
                       num_steps=50, guidance_scale=3.0, seed=None,
                       progress_callback=None):
    """Generate an image from audio via latent diffusion.

    audio_path=None runs Dream Mode: pure null conditioning sampled from the
    model's internalized distribution. With audio, classifier-free guidance
    blends conditioned and null noise predictions (guidance_scale<=1 disables).
    """
    device = next(unet.parameters()).device
    cond_dim = unet.cond_mlp[0].in_features
    null_cond = torch.zeros((1, cond_dim), device=device)

    if audio_path:
        conditioning = clap.embed_file(audio_path)
        use_cfg = guidance_scale > 1.0
    else:
        print("[pipeline] Dream Mode (null conditioning)")
        conditioning = null_cond
        use_cfg = False

    if seed is not None:
        torch.manual_seed(seed)
    latents = torch.randn((1, codec.latent_channels, codec.spatial_size, codec.spatial_size), device=device)

    total_train_steps = scheduler.num_train_timesteps
    inference_timesteps = np.linspace(total_train_steps - 1, 0, num_steps).astype(int)

    for i, t in enumerate(inference_timesteps):
        t_tensor = torch.tensor([t], device=device).long()

        noise_pred = unet(latents, t_tensor, conditioning)
        if use_cfg:
            noise_null = unet(latents, t_tensor, null_cond)
            noise_pred = noise_null + guidance_scale * (noise_pred - noise_null)

        latents = scheduler.step(noise_pred, t, latents)
        if progress_callback:
            progress_callback(int((i + 1) / num_steps * 100))

    image = codec.decode(latents)
    image = (image + 1) / 2
    return torch.clamp(image, 0, 1)
