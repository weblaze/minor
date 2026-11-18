"""Generator network of G-GAN (methods/gan/train_gan.py).

Moved unchanged from methods/gan/train_gan.py, so that bridges can load a trained generator through
abstraction.generators.GANGenerator. Parameter names are those of the checkpoints written before the move.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class UpBlock(nn.Module):
    def __init__(self, a, b):
        super().__init__()
        self.norm1, self.conv1 = nn.BatchNorm2d(a), nn.Conv2d(a, b, 3, padding=1)
        self.norm2, self.conv2 = nn.BatchNorm2d(b), nn.Conv2d(b, b, 3, padding=1)
        self.skip = nn.Conv2d(a, b, 1)

    def forward(self, x):
        h = self.conv1(F.interpolate(F.relu(self.norm1(x)), scale_factor=2, mode="nearest"))
        h = self.conv2(F.relu(self.norm2(h)))
        return h + self.skip(F.interpolate(x, scale_factor=2, mode="nearest"))


class Generator(nn.Module):
    def __init__(self, latent_dim, ch):
        super().__init__()
        widths = [ch * 8, ch * 8, ch * 4, ch * 2, ch, ch]
        self.first = nn.Linear(latent_dim, widths[0] * 16)
        self.blocks = nn.Sequential(*[UpBlock(a, b) for a, b in zip(widths[:-1], widths[1:])])
        self.norm = nn.BatchNorm2d(widths[-1])
        self.out = nn.Conv2d(widths[-1], 3, 3, padding=1)
        self.widths = widths

    def forward(self, z):
        x = self.first(z).view(len(z), self.widths[0], 4, 4)
        return torch.tanh(self.out(F.relu(self.norm(self.blocks(x)))))
