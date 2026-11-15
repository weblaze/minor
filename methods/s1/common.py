import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abstraction.generators import add_generator_argument
from abstraction.generators.vae import SpatialVAEGenerator
from abstraction.models.mapper import Mapper
from abstraction.pipeline import cli

CONFIG_PATH = Path(__file__).with_name("config.yaml")
GENERATOR = "spatial_vae"


def base_parser(description, generator=False):
    parser = cli.base_parser(description, CONFIG_PATH)
    return add_generator_argument(parser, GENERATOR) if generator else parser


def run_name(stem, smoke):
    return cli.run_name("s1", stem, smoke)


def build_vae(config):
    return SpatialVAEGenerator.build(config)


def latent_dim(generator):
    return math.prod(generator.latent_shape)


def code_dim(config):
    return config["audio_code"]["whiten_dim"] if config["audio_code"]["variant"] == "whitened" else 768


def build_mapper(config, generator):
    return Mapper(code_dim(config), latent_dim(generator), config["mapper"]["hidden"], config["mapper"]["depth"])


@torch.no_grad()
def encode_images(vae, dataset, device, batch_size=64):
    """Posterior means of every image of a dataset, on the CPU. `vae` is a SpatialVAE or a generator."""
    out = []
    for images in DataLoader(dataset, batch_size=batch_size):
        out.append(vae.encode(images.to(device))[0].float().cpu())
    return torch.cat(out)


def generate(mapper, generator, codes, code_mean, code_std):
    """Audio codes (n, d) -> (images, latents in units of generator.latent_scale)."""
    z = mapper((codes - code_mean) / code_std)
    z = z.view(-1, *generator.latent_shape)
    return generator.decode(z * generator.latent_scale), z
