import torch
import torch.nn.functional as F


def pairwise_distances(x, metric="cosine"):
    """Full (n, n) distance matrix between the rows of x."""
    x = x.flatten(1)
    if metric == "cosine":
        x = F.normalize(x, dim=1)
        return (1 - x @ x.T).clamp_min(0)
    if metric == "euclidean":
        sq = (x * x).sum(1)
        d2 = (sq[:, None] + sq[None, :] - 2 * x @ x.T).clamp_min(0)
        return (d2 + 1e-12).sqrt()
    raise ValueError(f"unknown metric {metric!r}")


def upper_triangle(d):
    i, j = torch.triu_indices(d.shape[0], d.shape[1], offset=1, device=d.device)
    return d[i, j]


def pearson(a, b, eps=1e-8):
    a = a - a.mean()
    b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + eps)


def distance_correlation_loss(d_source, d_target):
    """1 - Pearson correlation between the off-diagonal entries of two distance matrices."""
    return 1 - pearson(upper_triangle(d_source), upper_triangle(d_target))


def neighbour_kl_loss(d_source, d_target, temperature=0.5):
    """KL divergence between row-wise soft neighbour distributions of two distance matrices.

    Each matrix is divided by its mean off-diagonal value, which removes the
    scale of the space. Row i then defines p(j | i) proportional to
    exp(-d[i, j] / temperature) over j != i. The loss is the mean over rows of
    KL(p_source || p_target). It weights near neighbours, whereas the
    correlation loss is dominated by large distances.
    """
    n = d_source.shape[0]
    off = ~torch.eye(n, dtype=torch.bool, device=d_source.device)

    def log_neighbours(d):
        d = d / d[off].mean().clamp_min(1e-8)
        return F.log_softmax((-d / temperature).masked_fill(~off, float("-inf")), dim=1)

    log_p, log_q = log_neighbours(d_source), log_neighbours(d_target)
    kl = (log_p.exp() * (log_p - log_q).masked_fill(~off, 0.0)).sum(dim=1)
    return kl.mean()
