"""Paper 1 checks on the art collection (CPU, cached DINOv2 features only).

1. Exact and near duplicates across splits: for each val/test painting, the largest cosine similarity to a
   training painting; counts below cosine distances 0.001, 0.01, 0.05, 0.1; train-train twins.
2. FD reference for assignments of distinct real paintings: FD of n random distinct training paintings
   (n = 789, the number of test tracks) against the test paintings, over several draws. S2 assigns 789
   distinct training paintings, so its FD is read against this value, not against all 4666 (FD 113.3).

Usage: python analysis/paper1/art_checks.py [--draws 20] [--out runs/paper1_art_checks.json]
"""
import argparse
import json
from pathlib import Path

import numpy as np

from abstraction.data.splits import split_of
from abstraction.metrics.distribution import frechet_distance
from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="methods/f1/config.yaml")
    parser.add_argument("--draws", type=int, default=20)
    parser.add_argument("--n", type=int, default=789)
    parser.add_argument("--out", default="runs/paper1_art_checks.json")
    args = parser.parse_args()
    config = load_config(Path(args.config))
    with np.load(Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz") as d:
        names, feats = d["names"], d["features"].astype(np.float64)
    split = np.array([split_of(n, config["split"]["val_percent"], config["split"]["test_percent"]) for n in names])
    unit = feats / np.linalg.norm(feats, axis=1, keepdims=True)
    train = unit[split == "train"]
    out = {"n": {s: int((split == s).sum()) for s in ("train", "val", "test")}}
    for s in ("val", "test"):
        dist = 1 - (unit[split == s] @ train.T).max(axis=1)
        out[f"{s}_to_train"] = {str(t): int((dist < t).sum()) for t in (0.001, 0.01, 0.05, 0.1)}
        stems = {n.split("__", 1)[1] for n in names[split == "train"]}
        out[f"{s}_same_file_other_style"] = int(sum(n.split("__", 1)[1] in stems for n in names[split == s]))
    sim = train @ train.T
    np.fill_diagonal(sim, -1)
    out["train_with_train_twin_0.001"] = int((1 - sim.max(axis=1) < 0.001).sum())
    test_raw, train_raw = feats[split == "test"], feats[split == "train"]
    rng = np.random.default_rng(0)
    fds = [frechet_distance(train_raw[rng.choice(len(train_raw), args.n, replace=False)], test_raw)
           for _ in range(args.draws)]
    out["fd_random_distinct_training"] = {"n": args.n, "draws": args.draws, "mean": float(np.mean(fds)),
                                          "sd": float(np.std(fds, ddof=1)), "min": float(np.min(fds)),
                                          "max": float(np.max(fds))}
    out["fd_all_training"] = float(frechet_distance(train_raw, test_raw))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
