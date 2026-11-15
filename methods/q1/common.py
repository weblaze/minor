import json
import math
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from abstraction.generators import add_generator_argument
from abstraction.generators.vae import VectorVAEGenerator
from abstraction.models.mapper import Mapper
from abstraction.pipeline import cli

CONFIG_PATH = Path(__file__).with_name("config.yaml")
GENERATOR = "vector_vae"


def base_parser(description, generator=False):
    parser = cli.base_parser(description, CONFIG_PATH)
    return add_generator_argument(parser, GENERATOR) if generator else parser


def run_name(stem, smoke):
    return cli.run_name("q1", stem, smoke)


def build_vae(config):
    return VectorVAEGenerator.build(config)


def gate_passed(vae_path):
    """True, False, or None if no gate result exists next to the VAE checkpoint."""
    path = Path(vae_path).with_name("gate.json")
    if not path.exists():
        return None
    with open(path) as f:
        return bool(json.load(f)["passed"])


def build_mapper(config, code_dim, generator):
    return Mapper(code_dim, math.prod(generator.latent_shape), config["mapper"]["hidden"], config["mapper"]["depth"])


@torch.no_grad()
def encode_images(generator, dataset, device, batch_size=64):
    """Posterior means and log-variances of every image of a dataset, on the CPU."""
    mus, logvars = [], []
    for images in DataLoader(dataset, batch_size=batch_size):
        mu, logvar = generator.encode(images.to(device))
        mus.append(mu.float().cpu())
        logvars.append(logvar.float().cpu())
    return torch.cat(mus), torch.cat(logvars)


@torch.no_grad()
def features_of_latents(dino, generator, latents, device, batch_size=64):
    """Decode latents and return (DINOv2 features, first batch of images)."""
    features, first = [], None
    for start in range(0, len(latents), batch_size):
        images = generator.decode(latents[start:start + batch_size].to(device))
        features.append(dino(images).float().cpu())
        if first is None:
            first = images.cpu()
    return torch.cat(features), first


def code_statistics(train_codes):
    """Per-dimension mean and standard deviation of training codes (n_tracks, n_windows, d)."""
    flat = train_codes.reshape(-1, train_codes.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def generate(mapper, generator, codes, code_mean, code_std):
    """Audio codes (b, d) -> (images in [-1, 1], latents (b, latent_dim)).

    Q1 decodes the output of the mapping as it is; generator.latent_scale is not applied.
    """
    z = mapper((codes - code_mean) / code_std)
    return generator.decode(z.view(-1, *generator.latent_shape)), z


def principal_axes(x, k):
    """Mean, top-k principal axes (d, k) and standard deviations along them."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(eigval)[::-1][:k]
    return mean, eigvec[:, order], np.sqrt(np.maximum(eigval[order], 1e-12))
