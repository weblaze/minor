import torch

from abstraction.models.spatial_vae import SpatialVAE
from abstraction.models.vector_vae import VectorVAE

from .base import Generator

SPATIAL_SIDE = 16


def _frozen(module, state, device):
    module = module.to(device)
    module.load_state_dict(state["model"])
    module.eval()
    for p in module.parameters():
        p.requires_grad = False
    return module


class VAEGenerator(Generator):
    def __init__(self, vae, latent_shape, latent_scale):
        self.vae = vae
        self.latent_shape = latent_shape
        self.latent_scale = latent_scale

    def decode(self, z):
        return self.vae.decode(z)

    def encode(self, x):
        return self.vae.encode(x)


class SpatialVAEGenerator(VAEGenerator):
    """S1 generator: SpatialVAE (config section vae: latent_channels, base_channels).

    latent_scale is the standard deviation of the posterior means of the training images,
    stored in the checkpoint by methods/s1/train_vae.py.
    """

    @staticmethod
    def build(config):
        return SpatialVAE(config["vae"]["latent_channels"], config["vae"]["base_channels"])

    @classmethod
    def load(cls, path, config, device):
        state = torch.load(path, map_location=device, weights_only=True)
        vae = _frozen(cls.build(config), state, device)
        return cls(vae, (config["vae"]["latent_channels"], SPATIAL_SIDE, SPATIAL_SIDE), float(state["latent_std"]))


class VectorVAEGenerator(VAEGenerator):
    """Q1 generator: VectorVAE (config section vae: latent_dim, base_channels); latent_scale 1."""

    @staticmethod
    def build(config):
        return VectorVAE(config["vae"]["latent_dim"], config["vae"]["base_channels"])

    @classmethod
    def load(cls, path, config, device):
        state = torch.load(path, map_location=device, weights_only=True)
        vae = _frozen(cls.build(config), state, device)
        return cls(vae, (config["vae"]["latent_dim"],), 1.0)
