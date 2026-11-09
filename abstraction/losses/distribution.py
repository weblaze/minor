import torch
import torch.nn.functional as F


def sliced_wasserstein(x, y, n_projections=256):
    """Squared sliced 2-Wasserstein distance between two samples (rows of x and y).

    Each sample is projected on `n_projections` random unit directions; the
    one-dimensional distance is computed from sorted projections. Samples of
    unequal size are compared on the quantile grid of the smaller one.
    """
    x, y = x.flatten(1), y.flatten(1)
    directions = F.normalize(torch.randn(n_projections, x.shape[1], device=x.device, dtype=x.dtype), dim=1)
    px = (x @ directions.T).sort(dim=0).values
    py = (y @ directions.T).sort(dim=0).values
    if px.shape[0] != py.shape[0]:
        n = min(px.shape[0], py.shape[0])
        q = torch.linspace(0, 1, n, device=x.device, dtype=x.dtype)
        px = torch.quantile(px, q, dim=0)
        py = torch.quantile(py, q, dim=0)
    return ((px - py) ** 2).mean()
