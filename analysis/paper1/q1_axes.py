"""Paper 1: axis orientation and seed agreement of the Q1 mean-code runs (CPU, from saved features.npz).

Axes: top-3 eigenvectors of the covariance of window-averaged training-track mean codes and of the 4666 training
paintings (DINOv2 CLS), the convention of methods/f1/orientation.py. For each run: correlation of test-track
audio scores with image scores per axis. For the three q1 seeds: Pearson r between image distance matrices,
mean same-track and other-track cosine similarity.

Usage: python analysis/paper1/q1_axes.py [--kaggle ../../minor/runs/kaggle]
"""
import argparse
import itertools
import json
import sys

import numpy as np
from pathlib import Path
from abstraction.pipeline.data import load_layers, codes_of
from abstraction.data.splits import split_of
from abstraction.utils.config import load_config
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kaggle", default="../../minor/runs/kaggle")
parser.add_argument("--out", default="runs/paper1_q1_axes.json")
args = parser.parse_args()
K = Path(args.kaggle)
result = {}
runs = {"q1 s0": K/"abstraction-code-swap/runs/q1_eval_q1_meancode_seed0_20261112T174353Z",
        "q1 s1": K/"abstraction-q1-mean-seeds12/runs/q1_eval_q1_meancode_seed1_20261114T004053Z",
        "q1 s2": K/"abstraction-q1-mean-seeds12/runs/q1_eval_q1_meancode_seed2_20261114T002241Z",
        "shuffled": K/"abstraction-q1-mean-controls/runs/q1_eval_shuffled_meancode_seed0_20261114T003058Z",
        "no_feat": K/"abstraction-q1-mean-controls/runs/q1_eval_no_feat_meancode_seed0_20261114T004033Z"}
def axes_of(x, k=3):
    m = x.mean(0); w, v = np.linalg.eigh(np.cov(x - m, rowvar=False)); return m, v[:, np.argsort(w)[::-1][:k]]
config = load_config(Path("methods/f1/config.yaml"))
ids, layers, splits = load_layers(config)
codes, _ = codes_of(layers, splits, "mean", 128, None)
tracks = codes.mean(axis=1).astype(np.float64)
cm, ca = axes_of(tracks[splits == "train"])
with np.load(Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz") as d:
    names, art = d["names"], d["features"].astype(np.float64)
asp = np.array([split_of(n, 10, 10) for n in names])
am, aa = axes_of(art[asp == "train"])
pos = {t: i for i, t in enumerate(ids)}
feats = {}
for name, run in runs.items():
    d = np.load(run / "features.npz")
    idx = [pos[t] for t in d["track_ids"]]
    a = (tracks[idx] - cm) @ ca; im = (d["image_features"].astype(np.float64) - am) @ aa
    result[name] = {"axis_corr": [float(np.corrcoef(a[:, k], im[:, k])[0, 1]) for k in range(3)],
                    "codes_match_npz": bool(np.allclose(tracks[idx], d["audio_codes"], atol=1e-2))}
    f = d["image_features"].astype(np.float64); feats[name] = f / np.linalg.norm(f, axis=1, keepdims=True)
iu = np.triu_indices(789, 1)
for a, b in itertools.combinations(["q1 s0", "q1 s1", "q1 s2"], 2):
    A, B = feats[a], feats[b]; S = A @ B.T; n = len(S)
    r = np.corrcoef((1 - A @ A.T)[iu], (1 - B @ B.T)[iu])[0, 1]
    result[f"{a} vs {b}"] = {"image_distance_r": float(r), "same_track_cos": float(np.trace(S) / n),
                             "other_track_cos": float((S.sum() - np.trace(S)) / (n * (n - 1)))}
Path(args.out).write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
