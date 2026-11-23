import importlib.util
import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF

from abstraction.generators import add_generator_argument
from abstraction.models.mapper import Mapper
from abstraction.pipeline import cli

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DINO_DIM = 384


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


S1_DIR = Path(__file__).parents[1] / "s1"
s1_common = _load("s1_common", S1_DIR / "common.py")
s1_data = _load("s1_data", S1_DIR / "data.py")
build_mapper, generate, code_dim = s1_common.build_mapper, s1_common.generate, s1_common.code_dim
load_codes = s1_data.load_codes


def base_parser(description):
    return add_generator_argument(cli.base_parser(description, CONFIG_PATH), "gan")


def run_name(stem, smoke):
    return cli.run_name("s3", stem, smoke)


class Reader(nn.Module):
    """Both sides of the contrastive game: images (DINOv2 features) and audio codes into one unit sphere."""

    def __init__(self, audio_dim, embed_dim, hidden, image_dim=DINO_DIM):
        super().__init__()
        self.image = Mapper(image_dim, embed_dim, hidden, 2)
        self.audio = Mapper(audio_dim, embed_dim, hidden, 2)

    def embed_images(self, features):
        return F.normalize(self.image(features), dim=1)

    def embed_audio(self, codes):
        return F.normalize(self.audio(codes), dim=1)


def augment(images, aug, generator):
    """Random resized crop (aspect kept), additive noise and, for a random part of the batch, Gaussian blur.
    No flip and no colour change. Differentiable with respect to `images`."""
    n, device = images.shape[0], images.device
    low, high = aug["crop_scale"]
    scale = (low + (high - low) * torch.rand(n, generator=generator)).sqrt().to(device)
    shift = ((1 - scale.cpu())[:, None] * (2 * torch.rand(n, 2, generator=generator) - 1)).to(device)
    theta = torch.zeros(n, 2, 3, device=device)
    theta[:, 0, 0] = theta[:, 1, 1] = scale
    theta[:, :, 2] = shift
    grid = F.affine_grid(theta, list(images.shape), align_corners=False)
    out = F.grid_sample(images, grid, mode="bilinear", padding_mode="reflection", align_corners=False)
    if aug["blur_prob"] > 0 and aug["blur_sigma"] > 0:
        chosen = (torch.rand(n, generator=generator) < aug["blur_prob"]).to(device)
        if chosen.any():
            sigma = aug["blur_sigma"]
            size = 2 * math.ceil(2 * sigma) + 1
            blurred = TF.gaussian_blur(out, [size, size], [sigma, sigma])
            out = torch.where(chosen[:, None, None, None], blurred, out)
    if aug["noise_std"] > 0:
        noise = torch.randn(out.shape, generator=generator).to(device)
        out = out + aug["noise_std"] * noise
    return out.clamp(-1, 1)


def info_nce(image_embed, audio_embed, temperature):
    logits = image_embed @ audio_embed.T / temperature
    target = torch.arange(len(logits), device=logits.device)
    return 0.5 * (F.cross_entropy(logits, target) + F.cross_entropy(logits.T, target))


class TwinPenalty:
    """Repetition penalty. Each image's nearest neighbour among the batch and a bank of recent (detached) images,
    by DINOv2 cosine; squared hinge above `margin`. Matches the twin share of methods/s1/evaluate.py, which a
    batch alone rarely triggers (batch-only version, 2026-11-23: zero after a few steps)."""

    def __init__(self, size, margin):
        self.size, self.margin, self.bank = size, margin, None

    def __call__(self, features):
        unit = F.normalize(features.float(), dim=-1)
        sim = unit @ unit.T
        sim = sim.masked_fill(torch.eye(len(unit), dtype=torch.bool, device=unit.device), -1)
        nearest = sim.max(1).values
        if self.bank is not None:
            nearest = torch.maximum(nearest, (unit @ self.bank.T).max(1).values)
        new = unit.detach() if self.bank is None else torch.cat([unit.detach(), self.bank])
        self.bank = new[:self.size]
        return F.relu(nearest - self.margin).pow(2).mean(), (nearest > self.margin).float().mean()
