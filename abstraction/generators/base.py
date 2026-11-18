import torch


class Generator:
    """Frozen image generator used by a bridge in phase 2 and later.

    latent_shape  shape of one latent, without the batch dimension
    latent_scale  factor applied to the output of a mapping before decoding; a bridge that
                  works in units of the latent standard deviation multiplies by it
    has_encoder   whether encode() is available
    decode(z)     latents (b, *latent_shape) -> images (b, 3, h, w) in [-1, 1]
    encode(x)     images -> (posterior mean, log-variance), each (b, *latent_shape); only for
                  generators with an encoder. Bridges that compare mapped latents with latents
                  of real images (S1 L_dist, Q1 L_lat, the reconstruction references of the
                  evaluation scripts and the Q1 gate) need it.
    prior_sample(n, generator)  n samples of N(0, I) in latent shape, on the CPU. Bridges use them in
                  place of latents of real images when the generator has no encoder (decision of
                  2026-11-18 for G-GAN).
    """

    latent_shape = ()
    latent_scale = 1.0
    has_encoder = True

    @classmethod
    def load(cls, path, config, device):
        raise NotImplementedError

    def decode(self, z):
        raise NotImplementedError

    def encode(self, x):
        raise NotImplementedError(f"{type(self).__name__} has no encoder")

    def prior_sample(self, n, generator=None):
        return torch.randn(n, *self.latent_shape, generator=generator)
