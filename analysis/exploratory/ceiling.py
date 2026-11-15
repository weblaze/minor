"""Exploratory: what limits r of the painting output.
(1) interpolation between the direct output and its nearest painting; (2) galleries of increasing size
sampled from a Gaussian fitted to the art features; (3) average of the k nearest paintings;
(4) displacement replaced by a random displacement of the same length."""
import sys
import numpy as np, torch
sys.path.insert(0, "methods/f1")
import common as f1, realism
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix, knn_overlap
from abstraction.metrics.distribution import frechet_distance

dev = torch.device("cuda")
cfg = load_config("methods/f1/config.yaml")
_, layers, splits = f1.load_layers(cfg)
_, art, art_splits = f1.load_art(cfg, dev)
gal, test_art = art[art_splits == "train"], art[art_splits == "test"]
gallery = torch.from_numpy(gal).to(dev)
up = lambda m: m[np.triu_indices(len(m), 1)]
N = lambda x: torch.nn.functional.normalize(x, dim=1)
mu = gal.mean(0); cov = np.cov(gal - mu, rowvar=False)
L = np.linalg.cholesky(cov + 1e-6 * np.eye(len(cov)))
print("FD of 4666 Gaussian samples to test art: %.0f (real training art 113)" % frechet_distance((mu + np.random.default_rng(0).normal(size=(4666, 384)) @ L.T).astype(np.float32), test_art))
for code in ["mean", "standardised"]:
    codes, _ = f1.codes_of(layers, splits, code, 128)
    train = torch.from_numpy(codes[splits == "train"]).to(dev)
    test = torch.from_numpy(codes[splits == "test"].mean(axis=1)).to(dev)
    da_m = distance_matrix(test.cpu().numpy()); da = up(da_m)
    fwd = realism.fit({**realism.DEFAULT}, cfg, train, [], gallery, dev, cfg["mapper"]["steps"], 0)
    with torch.no_grad():
        out, _ = fwd(test)
        sim = N(out) @ N(gallery).T
        top = sim.topk(16, dim=1).indices
    def sc(x):
        d = distance_matrix(x)
        return "r %.3f kNN %.3f FD %.0f" % (np.corrcoef(da, up(d))[0, 1], knn_overlap(da_m, d, 10)[0], frechet_distance(x, test_art))
    o = out.cpu().numpy(); p = gal[top[:, 0].cpu().numpy()]
    print(f"== {code}")
    for t in [0, 0.25, 0.5, 0.75, 1.0]:
        print("  interpolate t=%.2f (0 = direct output, 1 = nearest painting):" % t, sc(o + t * (p - o)))
    for k in [2, 4, 8, 16]:
        print("  mean of %2d nearest paintings:" % k, sc(gal[top[:, :k].cpu().numpy()].mean(1)))
    rng = np.random.default_rng(0)
    disp = p - o
    rnd = rng.normal(size=disp.shape) @ L.T
    rnd = rnd / np.linalg.norm(rnd, axis=1, keepdims=True) * np.linalg.norm(disp, axis=1, keepdims=True)
    print("  direct output + random displacement of the same length (art covariance):", sc(o + rnd))
    print("  displacement length / output length: %.2f; cosine between displacement and output: %.2f"
          % (np.mean(np.linalg.norm(disp, axis=1) / np.linalg.norm(o, axis=1)), np.mean((disp * o).sum(1) / np.linalg.norm(disp, axis=1) / np.linalg.norm(o, axis=1))))
    for n in [4666, 46660, 466600]:
        best = torch.full((len(o),), -2.0, device=dev); pick = torch.zeros(len(o), 384, device=dev)
        g = torch.Generator(device="cpu").manual_seed(1)
        for start in range(0, n, 46660):
            m = min(46660, n - start)
            chunk = (torch.from_numpy(mu) + torch.randn(m, 384, generator=g) @ torch.from_numpy(L.T).float()).to(dev)
            s = N(out) @ N(chunk).T
            v, i = s.max(1)
            better = v > best
            pick[better] = chunk[i[better]]; best[better] = v[better]
        print("  Gaussian gallery of %6d: displacement %.3f |" % (n, float(1 - best.mean())), sc(pick.cpu().numpy()), flush=True)
