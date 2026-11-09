import torch.nn as nn


class _Block(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.residual = in_dim == out_dim

    def forward(self, x):
        return x + self.net(x) if self.residual else self.net(x)


class Mapper(nn.Module):
    """Deterministic MLP from an audio embedding to a generator latent.

    `depth` blocks of Linear, LayerNorm, GELU with a residual connection
    wherever input and output width agree, followed by a linear output layer.
    """

    def __init__(self, in_dim, out_dim, hidden=1024, depth=3):
        super().__init__()
        dims = [in_dim] + [hidden] * depth
        self.blocks = nn.Sequential(*[_Block(a, b) for a, b in zip(dims[:-1], dims[1:])])
        self.out = nn.Linear(hidden, out_dim)

    def forward(self, x):
        return self.out(self.blocks(x))
