"""Exploratory: direct assignment of test tracks to training paintings that maximises the correlation
between audio distances and painting-feature distances. No mapping network; an upper reference for
how much of a code's distance structure real paintings can carry."""
import sys
import numpy as np, torch
sys.path.insert(0, "methods/f1")
import common as f1
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix

dev = torch.device("cuda")
cfg = load_config("methods/f1/config.yaml")
_, layers, splits = f1.load_layers(cfg)
_, art, art_splits = f1.load_art(cfg, dev)
gallery, test_art = art[art_splits == "train"], art[art_splits == "test"]
g = torch.from_numpy(gallery).to(dev)
g = torch.nn.functional.normalize(g, dim=1)
Dg = 1 - g @ g.T

def pearson(a, b):
    a = a - a.mean(); b = b - b.mean()
    return (a * b).sum() / (a.norm() * b.norm() + 1e-8)

for name, variant in [("mean", "mean"), ("standardised", "standardised"), ("whitened", "whitened")]:
    codes, _ = f1.codes_of(layers, splits, variant, 128)
    test = codes[splits == "test"].mean(axis=1)
    Da = torch.from_numpy(distance_matrix(test)).float().to(dev)
    n = len(test)
    iu = torch.triu_indices(n, n, 1, device=dev)
    torch.manual_seed(0)
    logits = (0.01 * torch.randn(n, len(gallery), device=dev)).requires_grad_()
    opt = torch.optim.Adam([logits], lr=0.05)
    steps = 3000
    for step in range(steps):
        temp = 1.0 * (0.02 / 1.0) ** (step / steps)
        P = torch.softmax(logits / temp, dim=1)
        Dp = P @ Dg @ P.T
        loss = 1 - pearson(Da[iu[0], iu[1]], Dp[iu[0], iu[1]])
        opt.zero_grad(); loss.backward(); opt.step()
    hard = logits.argmax(1).cpu().numpy()
    s = f1.quick_scores(test, gallery[hard], test_art, 10)
    print(name, "soft r %.3f | hard: r %.3f kNN %.3f FD %.0f PR %.1f distinct %d"
          % (1 - loss.item(), s["r"], s["knn_overlap"], s["frechet_distance"], s["participation_ratio"], len(np.unique(hard))), flush=True)
