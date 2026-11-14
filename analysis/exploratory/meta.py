"""Exploratory, evaluation only: what the leading axes of each audio code relate to in FMA metadata."""
import sys
import numpy as np, pandas as pd
REPO = r"C:\Users\Arunav Ganguly\Documents\GitHub\minor"
META = sys.argv[1]
sys.path.insert(0, REPO + r"\methods\q1")
from abstraction.utils.config import load_config
from data import apply_code_transform, fit_code_transform, load_layers
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import cross_val_predict

cfg = load_config(REPO + r"\methods\q1\config.yaml")
cfg["paths"]["mert_embeddings"] = REPO + r"\runs\kaggle\abstraction-features\mert_embeddings"
ids, layers, splits = load_layers(cfg)
tid = np.array([int(i) for i in ids])
codes = {}
for v in ["mean", "standardised", "whitened"]:
    t = fit_code_transform(layers[splits == "train"], v)
    codes[v] = apply_code_transform(layers, t).mean(axis=1)
del layers

tracks = pd.read_csv(META + "/tracks.csv", index_col=0, header=[0, 1]).loc[tid]
feats = pd.read_csv(META + "/features.csv", index_col=0, header=[0, 1, 2]).loc[tid]
echo = pd.read_csv(META + "/echonest.csv", index_col=0, header=[0, 1, 2])
genre = tracks[("track", "genre_top")].astype(str).values
artist = tracks[("artist", "id")].values
print("tracks", len(tid), "genres", dict(pd.Series(genre).value_counts()))

desc = {}
for f in ["rmse", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr"]:
    desc[f + " mean"] = feats[(f, "mean", "01")].values
    desc[f + " std"] = feats[(f, "std", "01")].values
for k in range(1, 8):
    desc[f"spectral_contrast{k} mean"] = feats[("spectral_contrast", "mean", f"{k:02d}")].values
for k in range(1, 6):
    desc[f"mfcc{k} mean"] = feats[("mfcc", "mean", f"{k:02d}")].values
desc = pd.DataFrame(desc)
audio = echo["echonest"]["audio_features"]
in_echo = np.isin(tid, audio.index.values)
print("tracks with Echo Nest attributes:", int(in_echo.sum()))
low = np.column_stack([feats[(f, s)].values for f in ["rmse", "spectral_centroid", "spectral_bandwidth", "spectral_rolloff", "zcr", "spectral_contrast", "mfcc", "chroma_cens", "tonnetz"] for s in ["mean", "std"]])
low = (low - low.mean(0)) / (low.std(0) + 1e-9)
print("low-level descriptor matrix", low.shape)

def eta2(x, labels):
    groups = pd.Series(x).groupby(labels)
    return float((groups.size() * (groups.mean() - x.mean()) ** 2).sum() / ((x - x.mean()) ** 2).sum())

def knn_rates(x, k=10):
    x = x / np.linalg.norm(x, axis=1, keepdims=True)
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    nn = np.argpartition(-sim, k, axis=1)[:, :k]
    g = (genre[nn] == genre[:, None]).mean()
    a = (artist[nn] == artist[:, None]).mean()
    return g, a

train = splits == "train"
for v, x in codes.items():
    mu = x[train].mean(0)
    eigval, eigvec = np.linalg.eigh(np.cov((x[train] - mu), rowvar=False))
    order = np.argsort(eigval)[::-1]
    eigval, eigvec = eigval[order], eigvec[:, order]
    pcs = (x - mu) @ eigvec[:, :10]
    g, a = knn_rates(x)
    print(f"\n=== {v}: 10-NN same-genre rate {g:.3f} (chance 0.125), same-artist rate {a:.3f}")
    if v == "whitened":
        continue
    print("axis | var frac | genre eta2 | R2 from low-level acoustics (5-fold ridge) | strongest single descriptors (Spearman) | Echo Nest (Spearman)")
    for i in range(6):
        p = pcs[:, i]
        pred = cross_val_predict(RidgeCV(alphas=np.logspace(-1, 3, 9)), low, p, cv=5)
        r2 = 1 - ((p - pred) ** 2).sum() / ((p - p.mean()) ** 2).sum()
        rho = desc.corrwith(pd.Series(p), method="spearman").sort_values(key=abs, ascending=False)[:3]
        e = audio.loc[tid[in_echo]].reset_index(drop=True).corrwith(pd.Series(p[in_echo]), method="spearman").sort_values(key=abs, ascending=False)[:3]
        print("%d | %.3f | %.2f | %.2f | %s | %s" % (i + 1, eigval[i] / eigval.sum(), eta2(p, genre), r2,
              ", ".join(f"{k} {val:+.2f}" for k, val in rho.items()), ", ".join(f"{k} {val:+.2f}" for k, val in e.items())))
    means = pd.DataFrame(pcs[:, :3]).groupby(genre).mean().round(2)
    print("genre means on axes 1-3 (code units; axis std: %s)" % np.round(np.sqrt(eigval[:3]), 2))
    print(means.to_string())
