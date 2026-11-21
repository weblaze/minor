import math
from pathlib import Path

import numpy as np
import torch

from abstraction.data.datasets import ImageDataset
from abstraction.data.splits import split_of
from abstraction.pipeline import cli
from abstraction.pipeline.features import build_dino, features_of_dataset

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DINO_DIM = 384


def base_parser(description):
    return cli.base_parser(description, CONFIG_PATH,
                           smoke_help="synthetic codes and features, small problem; checks the code path only")


def run_name(stem, smoke):
    return cli.run_name("s2", stem, smoke)


def load_art(config, device, smoke=False):
    """DINOv2 features of every painting: (file names, features (n, 384), split name per painting).

    Same cache file as F1 (paths.dino_art_features).
    """
    if smoke:
        rng = np.random.default_rng(1)
        centres = rng.normal(size=(6, DINO_DIM))
        features = (centres[rng.integers(6, size=400)] + 0.7 * rng.normal(size=(400, DINO_DIM))).astype(np.float32)
        names = np.array([f"synthetic_{i:04d}.png" for i in range(400)])
    else:
        cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
        if not cache.exists():
            dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
            features = features_of_dataset(build_dino(config, device, False), dataset, device)
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, names=np.asarray(dataset.image_files), features=features.numpy())
        with np.load(cache) as data:
            names, features = data["names"], data["features"]
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    splits = np.array([split_of(name, val, test) for name in names])
    return names, features, splits


def cosine_distances(x):
    """Cosine distance matrix of the rows of x, divided by its mean off-diagonal value."""
    unit = torch.nn.functional.normalize(x, dim=1)
    d = (1 - unit @ unit.T).clamp_min(0)
    d.fill_diagonal_(0)
    return d / (d.sum() / (len(d) * (len(d) - 1)))


def sinkhorn_log(cost, log_p, log_q, eps, iterations):
    """Entropic optimal transport plan for `cost` with marginals exp(log_p), exp(log_q) (log domain)."""
    f = torch.zeros_like(log_p)
    g = torch.zeros_like(log_q)
    k = -cost / eps
    for _ in range(iterations):
        f = log_p - torch.logsumexp(k + g[None, :], dim=1)
        g = log_q - torch.logsumexp(k + f[:, None], dim=0)
    return torch.exp(k + f[:, None] + g[None, :])


def gromov_wasserstein(d_a, d_b, eps, outer, inner, generator=None, tol=1e-7, eps_start=None, decay=0.95,
                       p=None, q=None, init=None):
    """Entropic Gromov-Wasserstein coupling between two metric spaces (square loss), with uniform weights
    unless the point weights `p`, `q` (each summing to 1) are given.

    Alternates the linearisation L(T) = const - 2 D_a T D_b with an entropic transport step
    (Peyre, Cuturi and Solomon, ICML 2016). The regularisation starts at `eps_start` (default `eps`) and is
    multiplied by `decay` at each step until it reaches `eps`; the stopping test applies only at `eps`.
    With `generator`, the first coupling is the transport plan of a random cost (a random restart);
    otherwise the independent coupling p q^T. `init` gives the first coupling directly.
    Returns the coupling and the objective sum_ijkl (D_a[i,k] - D_b[j,l])^2 T_ij T_kl.
    """
    n, m = len(d_a), len(d_b)
    device = d_a.device
    p = torch.full((n,), 1.0 / n, device=device, dtype=d_a.dtype) if p is None else p
    q = torch.full((m,), 1.0 / m, device=device, dtype=d_a.dtype) if q is None else q
    log_p, log_q = p.log(), q.log()
    constant = (d_a ** 2 @ p)[:, None] + (d_b ** 2 @ q)[None, :]
    if init is not None:
        coupling = init
    elif generator is None:
        coupling = p[:, None] * q[None, :]
    else:
        noise = torch.rand(n, m, generator=generator).to(device=device, dtype=d_a.dtype)
        coupling = sinkhorn_log(noise, log_p, log_q, 0.1, inner)
    previous = math.inf
    current = eps_start or eps
    for _ in range(outer):
        linear = constant - 2 * d_a @ coupling @ d_b
        value = float((linear * coupling).sum())
        if current <= eps and abs(previous - value) < tol:
            break
        previous = value
        coupling = sinkhorn_log(linear, log_p, log_q, current, inner)
        current = max(eps, current * decay)
    value = float(((constant - 2 * d_a @ coupling @ d_b) * coupling).sum())
    return coupling, value
