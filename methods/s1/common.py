import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from abstraction.encoders.dino import DinoFeatures
from abstraction.models.mapper import Mapper
from abstraction.models.spatial_vae import SpatialVAE
from abstraction.utils.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")
LATENT_SIDE = 16
CODE_DIM = 768


def base_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="appended to the run directory name")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data, randomly initialised DINOv2, few steps; checks the code path only")
    return parser


def setup(args):
    config = load_config(args.config)
    if args.seed is not None:
        config["system"]["seed"] = args.seed
    if args.tag:
        config["system"]["tag"] = args.tag
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    return config, device


def build_dino(config, device, smoke):
    if smoke:
        return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"], pretrained=False, depth=2).to(device)
    return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)


def build_vae(config):
    return SpatialVAE(config["vae"]["latent_channels"], config["vae"]["base_channels"])


def load_vae(path, config, device):
    """Returns the frozen VAE and the standard deviation of its training latents."""
    state = torch.load(path, map_location=device, weights_only=True)
    vae = build_vae(config).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae, float(state["latent_std"])


def latent_dim(config):
    return config["vae"]["latent_channels"] * LATENT_SIDE * LATENT_SIDE


def build_mapper(config):
    return Mapper(CODE_DIM, latent_dim(config), config["mapper"]["hidden"], config["mapper"]["depth"])


@torch.no_grad()
def encode_images(vae, dataset, device, batch_size=64):
    """Posterior means of every image of a dataset, on the CPU."""
    out = []
    for images in DataLoader(dataset, batch_size=batch_size):
        out.append(vae.encode(images.to(device))[0].float().cpu())
    return torch.cat(out)


@torch.no_grad()
def features_of_dataset(dino, dataset, device, batch_size=64, transform=None):
    """DINOv2 features of every image of a dataset, optionally after `transform`."""
    out = []
    for images in DataLoader(dataset, batch_size=batch_size):
        images = images.to(device)
        if transform is not None:
            images = transform(images)
        out.append(dino(images).float().cpu())
    return torch.cat(out)


def generate(mapper, vae, latent_std, codes, code_mean, code_std, config):
    """Audio codes (n, 768) -> (images, latents in units of latent_std)."""
    z = mapper((codes - code_mean) / code_std)
    z = z.view(-1, config["vae"]["latent_channels"], LATENT_SIDE, LATENT_SIDE)
    return vae.decode(z * latent_std), z
