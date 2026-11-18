"""Orientation of the leading axes in the S2 assignments, with the axis convention of methods/f1/orientation.py.

S2 runs before commit a677747 computed `corr` with an SVD, whose axis signs need not match F1. This script
recomputes, for every restart of every code in an S2 run directory, the correlation between the test tracks'
scores on principal axis k of the audio code (fitted on training tracks, mean over windows) and the scores
of their assigned paintings on principal axis k of the training paintings, using the eigh convention of F1.

Usage:
  python analysis/exploratory/s2_orientation.py runs/s2_eval_match_<...>
"""
import json
import sys
from pathlib import Path

import numpy as np

from abstraction.pipeline.data import codes_of, load_layers
from abstraction.data.splits import split_of
from abstraction.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]
N_AXES = 3


def axes_of(x, k):
    mean = x.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(eigval)[::-1][:k]
    return mean, eigvec[:, order]


def main():
    run = Path(sys.argv[1])
    config = load_config(ROOT / "methods" / "s2" / "config.yaml")
    ids, layers, splits = load_layers(config)
    cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
    with np.load(cache) as data:
        names, art = data["names"], data["features"]
    art_splits = np.array([split_of(n, config["split"]["val_percent"], config["split"]["test_percent"]) for n in names])
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    position = {str(n): i for i, n in enumerate(gallery_names)}
    art_mean, art_axes = axes_of(gallery, N_AXES)
    out = {}
    for path in sorted(run.glob("assignments_*.npz")):
        code = path.stem.removeprefix("assignments_")
        codes, _ = codes_of(layers, splits, code, config["audio_code"]["whiten_dim"], None)
        tracks = codes.mean(axis=1)
        code_mean, code_axes = axes_of(tracks[splits == "train"], N_AXES)
        audio_scores = (tracks[splits == "test"] - code_mean) @ code_axes
        with np.load(path) as data:
            assert list(data["track_ids"]) == list(ids[splits == "test"])
            picks = data["painting_names"]
        rows = []
        for restart, names_row in enumerate(picks):
            painting_scores = (gallery[[position[str(n)] for n in names_row]] - art_mean) @ art_axes
            rows.append([float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)])
        rows = np.array(rows)
        out[code] = {"corr": rows.round(4).tolist(),
                     "positive_counts": [int((rows[:, k] > 0).sum()) for k in range(N_AXES)]}
        print(code, "positive counts per axis", out[code]["positive_counts"], "mean corr", rows.mean(axis=0).round(3))
    (run / "orientation_f1_convention.json").write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
