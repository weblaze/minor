import os

import torch

from abstraction.models.image_vae import ImageVAE
from abstraction.utils.checkpoints import load_checkpoint

SD_SCALING_FACTOR = 0.18215


class OwnVAECodec:
    """Adapter over the project's ImageVAE: 8ch x 16x16 latents at 128px."""

    def __init__(self, checkpoint_path, latent_channels, device):
        self.device = device
        self.latent_channels = latent_channels
        self.spatial_size = 16
        self.vae = ImageVAE(latent_channels=latent_channels).to(device)
        if checkpoint_path and os.path.exists(checkpoint_path):
            load_checkpoint(self.vae, checkpoint_path, device=device)
            print(f"[codec] loaded ImageVAE from {checkpoint_path}")
        else:
            print(f"[codec] WARNING: no ImageVAE checkpoint at {checkpoint_path} — using untrained weights")
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images):
        mu, logvar, _ = self.vae.encode(images)
        return self.vae.reparameterize(mu, logvar)

    @torch.no_grad()
    def decode(self, latents):
        return self.vae.decode(latents)


class SDVAECodec:
    """Adapter over the pretrained Stable Diffusion VAE: 4ch x 16x16 latents at 128px."""

    def __init__(self, device):
        from diffusers import AutoencoderKL

        self.device = device
        self.latent_channels = 4
        self.spatial_size = 16
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        self.dtype = dtype
        self.vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse", torch_dtype=dtype).to(device)
        self.vae.enable_slicing()
        self.vae.eval()
        for p in self.vae.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def encode(self, images):
        posterior = self.vae.encode(images.to(self.dtype)).latent_dist
        return (posterior.sample() * SD_SCALING_FACTOR).float()

    @torch.no_grad()
    def decode(self, latents):
        images = self.vae.decode(latents.to(self.dtype) / SD_SCALING_FACTOR).sample
        return images.float()


def build_codec(config, device):
    kind = config["vae"]["kind"]
    if kind == "own":
        ckpt = config["paths"].get("image_vae_checkpoint")
        latent_channels = config["vae"].get("latent_channels", 8)
        return OwnVAECodec(ckpt, latent_channels, device)
    if kind == "sd":
        return SDVAECodec(device)
    raise ValueError(f"Unknown vae.kind: {kind!r} (expected 'own' or 'sd')")
