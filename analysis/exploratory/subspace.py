"""Exploratory: (1) structure of retrieved paintings measured in the leading principal axes of art features;
(2) relation between track genre and the WikiArt style of the retrieved painting (metadata, evaluation only)."""
import sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, "methods/f1")
import common as f1, realism
from abstraction.utils.config import load_config
from abstraction.metrics.structure import distance_matrix

dev = torch.device("cuda")
cfg = load_config("methods/f1/config.yaml")
ids, layers, splits = f1.load_layers(cfg)
names, art, art_splits = f1.load_art(cfg, dev)
gal_np, gal_names = art[art_splits == "train"], names[art_splits == "train"]
gallery = torch.from_numpy(gal_np).to(dev)
mu = gal_np.mean(0)
eigval, eigvec = np.linalg.eigh(np.cov(gal_np - mu, rowvar=False))
order = np.argsort(eigval)[::-1]; eigvec = eigvec[:, order]; eigval = eigval[order]
print("art variance fraction, cumulative at k=2,4,8,16,32,64:", np.round(np.cumsum(eigval)[[1, 3, 7, 15, 31, 63]] / eigval.sum(), 3))
tracks = pd.read_csv(sys.argv[1] + "/tracks.csv", index_col=0, header=[0, 1])
genre = tracks.loc[[int(i) for i in ids[splits == "test"]], ("track", "genre_top")].astype(str).values
styles = np.array([n.split("__")[0] for n in gal_names])
up = lambda m: m[np.triu_indices(len(m), 1)]
r = lambda a, b: float(np.corrcoef(a, b)[0, 1])
for code in ["mean", "standardised"]:
    codes, _ = f1.codes_of(layers, splits, code, 128)
    train = torch.from_numpy(codes[splits == "train"]).to(dev)
    test = torch.from_numpy(codes[splits == "test"].mean(axis=1)).to(dev)
    da = up(distance_matrix(test.cpu().numpy()))
    for name in ["swd", "soft_0.02_swd", "st_0.05_swd"]:
        spec = {**realism.DEFAULT, **realism.GRID[name]}
        fwd = realism.fit(spec, cfg, train, [], gallery, dev, cfg["mapper"]["steps"], 0)
        with torch.no_grad():
            out, q = fwd(test)
            near = (torch.nn.functional.normalize(q, dim=1) @ torch.nn.functional.normalize(gallery, dim=1).T).argmax(1).cpu().numpy()
        out = out.cpu().numpy(); paint = gal_np[near]
        row = []
        for k in [2, 4, 8, 16, 32, 64]:
            P = eigvec[:, :k]
            row.append("k=%d: painting %.3f direct %.3f" % (k, r(da, up(distance_matrix((paint - mu) @ P, metric="euclidean"))),
                                                          r(da, up(distance_matrix((out - mu) @ P, metric="euclidean")))))
        resid = (paint - mu) - ((paint - mu) @ eigvec[:, :16]) @ eigvec[:, :16].T
        print(code, name, "| full cosine painting %.3f |" % r(da, up(distance_matrix(paint))), " ; ".join(row),
              "| residual beyond 16 axes %.3f" % r(da, up(distance_matrix(resid, metric="euclidean"))), flush=True)
        if name == "swd":
            tab = pd.crosstab(genre, styles[near])
            chi2 = ((tab - np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum()) ** 2 / (np.outer(tab.sum(1), tab.sum(0)) / tab.values.sum())).values.sum()
            v = np.sqrt(chi2 / (tab.values.sum() * (min(tab.shape) - 1)))
            print("genre x style of retrieved painting, row fractions (Cramer V %.3f); gallery style share: %s"
                  % (v, dict(pd.Series(styles).value_counts(normalize=True).round(2))))
            print((tab.div(tab.sum(1), axis=0)).round(2).to_string(), flush=True)
