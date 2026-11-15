"""Exploratory. Feature-space mapping: which distribution term places outputs on the art distribution,
and which audio code can be embedded in it. Not part of the repository protocol."""
import sys, glob
import numpy as np, torch
from scipy.optimize import linear_sum_assignment
REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
sys.path.insert(0, REPO + r"\methods\q1")
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix, knn_overlap
from abstraction.metrics.distribution import frechet_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import pairwise_distances, distance_correlation_loss
from abstraction.models.mapper import Mapper
from data import apply_code_transform, fit_code_transform, load_layers

d = dict(np.load(sys.argv[1]))
cfg = load_config(REPO + r"\methods\q1\config.yaml")
cfg["paths"]["mert_embeddings"] = REPO + r"\runs\kaggle\abstraction-features\mert_embeddings"
ids, layers, splits = load_layers(cfg)
tr = fit_code_transform(layers[splits == "train"], "standardised")
for s in ["train", "test"]:
    d[f"std_{s}"] = apply_code_transform(layers[splits == s], tr)
del layers
dev = torch.device("cuda")
art_train = torch.from_numpy(d["art_train"]).to(dev)
art_test = d["art_test"]
up = lambda m: m[np.triu_indices(len(m), 1)]
r = lambda a, b: float(np.corrcoef(a, b)[0, 1])

def energy(x, y):
    return 2 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean()

def ot(x, y):
    cost = torch.cdist(x, y) ** 2
    row, col = linear_sum_assignment(cost.detach().cpu().numpy())
    return cost[row, col].mean()

TERMS = {"swd": lambda x, y: sliced_wasserstein(x, y, 256), "energy": energy, "ot": ot}

def run(code, term, lam, steps=4000, seed=0):
    torch.manual_seed(seed)
    trc = torch.from_numpy(d[f"{code}_train"]).to(dev)
    te = torch.from_numpy(d[f"{code}_test"].mean(1)).to(dev)
    m, s = trc.mean((0, 1)), trc.std((0, 1)) + 1e-6
    mapper = Mapper(trc.shape[-1], 384, 1024, 3).to(dev)
    opt = torch.optim.AdamW(mapper.parameters(), lr=1e-4)
    scale = art_train.std()
    for step in range(steps):
        t = torch.randperm(trc.shape[0], device=dev)[:256]
        codes = trc[t, torch.randint(trc.shape[1], (256,), device=dev)]
        out = mapper((codes - m) / s)
        real = art_train[torch.randperm(art_train.shape[0], device=dev)[:256]]
        loss = TERMS[term](out / scale, real / scale)
        if lam > 0:
            loss = loss + lam * distance_correlation_loss(pairwise_distances(codes), pairwise_distances(out))
        opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
    with torch.no_grad():
        out = mapper.eval()((te - m) / s)
        a = torch.nn.functional.normalize(art_train, dim=1)
        near = (torch.nn.functional.normalize(out, dim=1) @ a.T).argmax(1)
        snapped = art_train[near].cpu().numpy(); out = out.cpu().numpy()
    da = distance_matrix(te.cpu().numpy())
    res = []
    for f in (out, snapped):
        di = distance_matrix(f)
        res.append((r(up(da), up(di)), knn_overlap(da, di, 10)[0], frechet_distance(f, art_test), participation_ratio(f)))
    return res, int(near.unique().numel())

print("code  term   lam | direct: Mantel kNN FD PR | snapped to real paintings: Mantel kNN FD PR distinct", flush=True)
for code in ["mean", "std"]:
    for term in ["swd", "energy", "ot"]:
        for lam in [1.0] if term != "ot" else [0.0, 0.3, 1.0]:
            (a, b), n = run(code, term, lam)
            print("%-5s %-6s %-3s | %.3f %.3f %6.1f %5.1f | %.3f %.3f %6.1f %5.1f %d" % (code, term, lam, *a, *b, n), flush=True)
