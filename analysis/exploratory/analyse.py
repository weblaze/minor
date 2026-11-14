"""Exploratory analysis, not part of the repository protocol.

A. Structure of S1 and Q1 outputs measured with features other than DINOv2.
B. S1 structure after removing leading principal axes of the mean code.
C. Mapping audio codes directly into DINOv2 feature space (no decoder), with and without
   snapping to the nearest real training painting.
"""
import sys
import numpy as np
import torch

REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
sys.path.insert(0, REPO + r"\methods\q1")
from abstraction.metrics.structure import distance_matrix, knn_overlap
from abstraction.metrics.distribution import frechet_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import pairwise_distances, distance_correlation_loss
from abstraction.models.mapper import Mapper

d = np.load(sys.argv[1])
iu = None


def upper(m):
    return m[np.triu_indices(len(m), 1)]


def r(a, b):
    return float(np.corrcoef(a, b)[0, 1])


def eucl(x):
    x = np.asarray(x, dtype=np.float64)
    g = (x ** 2).sum(1)
    return np.sqrt(np.maximum(g[:, None] + g[None] - 2 * x @ x.T, 0))


def image_stats(x):
    """x: (n, 3, H, W) in [-1, 1]. Returns dict of simple feature sets."""
    x = torch.from_numpy(x.astype(np.float32))
    out = {}
    out["colour mean+std (6)"] = torch.cat([x.mean((2, 3)), x.std((2, 3))], 1).numpy()
    q = ((x * 0.5 + 0.5).clamp(0, 0.999) * 4).long()
    idx = (q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]).flatten(1)
    out["colour histogram (64)"] = torch.stack([torch.bincount(i, minlength=64).float() / i.numel() for i in idx]).numpy()
    out["layout 8x8 (192)"] = torch.nn.functional.adaptive_avg_pool2d(x, 8).flatten(1).numpy()
    grey = x.mean(1)
    power = torch.fft.fftshift(torch.fft.fft2(grey - grey.mean((1, 2), keepdim=True)).abs() ** 2, dim=(1, 2))
    h = grey.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(h) - h // 2, torch.arange(h) - h // 2, indexing="ij")
    rad = torch.sqrt((yy ** 2 + xx ** 2).float()).long().clamp(max=h // 2 - 1)
    bins = torch.stack([power[:, rad == k].mean(1) for k in range(1, h // 2)], 1)
    out["power spectrum (63)"] = torch.log(bins + 1e-8).numpy()
    return out


mean_test = d["mean_test"].mean(1)
white_test = d["white_test"].mean(1)
da_mean, da_white = upper(distance_matrix(mean_test)), upper(distance_matrix(white_test))

print("== A. structure measured with other features (Pearson r of distance matrices, 789 test tracks)")
print("%-24s | S1 vs mean code | Q1 vs whitened code" % "feature set")
s1, q1 = image_stats(d["s1_images"]), image_stats(d["q1_images"])
print("%-24s | %.3f | %.3f" % ("DINOv2 (training target)", r(upper(distance_matrix(d["s1_feats"])), da_mean),
                              r(upper(distance_matrix(d["q1_feats"])), da_white)))
for k in s1:
    print("%-24s | %.3f | %.3f" % (k, r(upper(eucl(s1[k])), da_mean), r(upper(eucl(q1[k])), da_white)))

print("== A2. spread of simple features relative to real test art (mean pairwise distance, generated / art)")
art = image_stats(d["art_test_images"])
for k in s1:
    a = upper(eucl(art[k])).mean()
    print("%-24s | S1 %.2f | Q1 %.2f" % (k, upper(eucl(s1[k])).mean() / a, upper(eucl(q1[k])).mean() / a))

print("== B. mean code: leading principal axes")
train = d["mean_train"].reshape(-1, d["mean_train"].shape[-1]).astype(np.float64)
mu = train.mean(0)
eigval, eigvec = np.linalg.eigh(np.cov(train - mu, rowvar=False))
order = np.argsort(eigval)[::-1]
eigval, eigvec = eigval[order], eigvec[:, order]
print("variance fraction of first axes:", np.round(eigval[:6] / eigval.sum(), 3))
ds1 = upper(distance_matrix(d["s1_feats"]))
dq1 = upper(distance_matrix(d["q1_feats"]))
for k in [0, 1, 2, 5, 10, 20]:
    x = mean_test - mu
    x = x - (x @ eigvec[:, :k]) @ eigvec[:, :k].T
    dk = upper(distance_matrix(x + (mu if k == 0 else 0)))
    print("remove %2d axes: r(S1 image dist, code dist) = %.3f | r(with full mean-code dist) = %.3f | r(with whitened) = %.3f"
          % (k, r(ds1, dk), r(dk, da_mean), r(dk, da_white)))
for k in [1, 2, 5, 10]:
    x = (mean_test - mu) @ eigvec[:, :k]
    print("only top %2d axes (euclid): r(with full mean-code cosine dist) = %.3f | r(S1 image dist) = %.3f"
          % (k, r(upper(eucl(x)), da_mean), r(ds1, upper(eucl(x)))))

print("== C. mapping into DINOv2 feature space, no decoder")
dev = torch.device("cuda")
art_train = torch.from_numpy(d["art_train"]).to(dev)
art_test = d["art_test"]
print("reference: FD(train art, test art) = %.1f" % frechet_distance(d["art_train"], art_test))


def run(code_name, lam, steps=4000, seed=0, snap=False):
    torch.manual_seed(seed)
    tr = torch.from_numpy(d[f"{code_name}_train"]).to(dev)
    te = torch.from_numpy(d[f"{code_name}_test"].mean(1)).to(dev)
    m, s = tr.mean((0, 1)), tr.std((0, 1)) + 1e-6
    mapper = Mapper(tr.shape[-1], art_train.shape[1], 1024, 3).to(dev)
    opt = torch.optim.AdamW(mapper.parameters(), lr=1e-4)
    scale = art_train.std()
    for step in range(steps):
        t = torch.randperm(tr.shape[0], device=dev)[:256]
        w = torch.randint(tr.shape[1], (256,), device=dev)
        codes = tr[t, w]
        out = mapper((codes - m) / s)
        real = art_train[torch.randint(art_train.shape[0], (256,), device=dev)]
        loss = sliced_wasserstein(out / scale, real / scale, 256)
        if lam > 0:
            loss = loss + lam * distance_correlation_loss(pairwise_distances(codes), pairwise_distances(out))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    with torch.no_grad():
        out = mapper.eval()((te - m) / s)
        if snap:
            a = torch.nn.functional.normalize(art_train, dim=1)
            nearest = (torch.nn.functional.normalize(out, dim=1) @ a.T).argmax(1)
            n_distinct = int(nearest.unique().numel())
            out = art_train[nearest]
        out = out.cpu().numpy()
    codes = te.cpu().numpy()
    da, di = distance_matrix(codes), distance_matrix(out)
    res = (r(upper(da), upper(di)), knn_overlap(da, di, 10)[0], frechet_distance(out, art_test),
           upper(di).mean(), participation_ratio(out))
    return res + ((n_distinct,) if snap else ())


print("%-9s %-5s %-5s | Mantel r | kNN | FD | mean pair dist (art 0.672) | PR (art 44.8) | distinct paintings" % ("code", "lam", "snap"))
for code_name in ["mean", "white"]:
    for lam in [0.0, 1.0, 3.0, 10.0]:
        for snap in [False, True]:
            res = run(code_name, lam, snap=snap)
            print("%-9s %-5s %-5s | %.3f | %.3f | %.1f | %.3f | %.1f | %s"
                  % (code_name, lam, snap, res[0], res[1], res[2], res[3], res[4], res[5] if snap else ""))
