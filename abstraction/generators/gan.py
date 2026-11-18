import torch

from abstraction.models.gan import Generator as GANNetwork

from .base import Generator


class GANGenerator(Generator):
    """G-GAN generator (methods/gan/train_gan.py): the averaged generator stored in `generator.pt`.

    The latent is the generator input, a vector with prior N(0, I); latent_scale 1. The generator has no
    encoder: bridges use samples of the prior where they would use latents of real images
    (`prior_sample`).
    """

    has_encoder = False

    def __init__(self, network, latent_dim):
        self.network = network
        self.latent_shape = (latent_dim,)
        self.latent_scale = 1.0

    @classmethod
    def load(cls, path, config, device):
        state = torch.load(path, map_location=device, weights_only=True)
        network = GANNetwork(state["latent_dim"], state["base_channels"]).to(device)
        network.load_state_dict(state["model"])
        network.eval()
        for p in network.parameters():
            p.requires_grad = False
        return cls(network, state["latent_dim"])

    def decode(self, z):
        return self.network(z)
