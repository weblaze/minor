import torch
import torch.nn as nn
import torch.nn.functional as F


def _norm(channels):
    return nn.GroupNorm(32, channels)


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.norm1, self.conv1 = _norm(in_ch), nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.norm2, self.conv2 = _norm(out_ch), nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        h = self.conv1(F.silu(self.norm1(x)))
        h = self.conv2(F.silu(self.norm2(h)))
        return self.skip(x) + h


class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2, mode="nearest"))


class SpatialVAE(nn.Module):
    """Convolutional VAE with a spatial latent: 3 x 128 x 128 -> C x 16 x 16.

    Residual blocks with GroupNorm; downsampling by strided convolution,
    upsampling by nearest-neighbour interpolation followed by a convolution.
    """

    def __init__(self, latent_channels=4, base_channels=64):
        super().__init__()
        c1, c2, c3 = base_channels, base_channels * 2, base_channels * 4
        self.encoder = nn.Sequential(
            nn.Conv2d(3, c1, 3, padding=1),
            ResBlock(c1, c1), nn.Conv2d(c1, c1, 3, stride=2, padding=1),
            ResBlock(c1, c2), nn.Conv2d(c2, c2, 3, stride=2, padding=1),
            ResBlock(c2, c3), nn.Conv2d(c3, c3, 3, stride=2, padding=1),
            ResBlock(c3, c3), _norm(c3), nn.SiLU(),
            nn.Conv2d(c3, 2 * latent_channels, 3, padding=1),
        )
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, c3, 3, padding=1),
            ResBlock(c3, c3), Upsample(c3),
            ResBlock(c3, c2), Upsample(c2),
            ResBlock(c2, c1), Upsample(c1),
            ResBlock(c1, c1), _norm(c1), nn.SiLU(),
            nn.Conv2d(c1, 3, 3, padding=1), nn.Tanh(),
        )

    def encode(self, x):
        mu, logvar = self.encoder(x).chunk(2, dim=1)
        return mu, logvar.clamp(-30.0, 20.0)

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = mu + torch.randn_like(mu) * (0.5 * logvar).exp()
        return self.decode(z), mu, logvar
