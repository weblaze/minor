"""Agreement between mappings trained with different seeds.

Input: two or more evaluation run directories (each holds features.npz from
evaluate.py) over the same test tracks. For every pair of runs:

  same_track_cosine    mean cosine similarity between the DINOv2 features of the
                       two images generated for the same track
  other_track_cosine   the same after pairing each track with a different track
                       (mean over all non-matching pairs)
  mantel_pearson       Mantel correlation between the two image distance matrices,
                       with a permutation p-value

same_track_cosine above other_track_cosine means the two seeds place a track in
similar regions of image-feature space. A high Mantel correlation with equal
same- and other-track cosine means the seeds agree on relative structure but not
on the images themselves; this is the expected outcome if the mapping is
determined only up to a transformation that preserves distances.

Usage:
  python methods/q1/compare_seeds.py runs/<eval run 1> runs/<eval run 2> [...] --out runs/seed_agreement.json
"""
import argparse
import itertools
import json
from pathlib import Path

import numpy as np

from abstraction.metrics.structure import distance_matrix, mantel


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="+")
    parser.add_argument("--out", required=True)
    parser.add_argument("--permutations", type=int, default=999)
    args = parser.parse_args()
    if len(args.runs) < 2:
        raise SystemExit("at least two evaluation runs are required")

    loaded = []
    for run in args.runs:
        data = np.load(Path(run) / "features.npz")
        features = data["image_features"].astype(np.float64)
        loaded.append((run, list(data["track_ids"]), features / np.linalg.norm(features, axis=1, keepdims=True)))

    pairs = []
    for (run_a, ids_a, a), (run_b, ids_b, b) in itertools.combinations(loaded, 2):
        if ids_a != ids_b:
            raise SystemExit(f"track ids differ between {run_a} and {run_b}")
        similarity = a @ b.T
        n = len(similarity)
        same = float(np.trace(similarity) / n)
        other = float((similarity.sum() - np.trace(similarity)) / (n * (n - 1)))
        r, p = mantel(distance_matrix(a), distance_matrix(b), "pearson", args.permutations)
        pairs.append({"run_a": str(run_a), "run_b": str(run_b), "n_tracks": n, "same_track_cosine": same,
                      "other_track_cosine": other, "mantel_pearson": r, "mantel_pearson_p": p})

    summary = {key: float(np.mean([pair[key] for pair in pairs]))
               for key in ("same_track_cosine", "other_track_cosine", "mantel_pearson")}
    with open(args.out, "w") as f:
        json.dump({"n_runs": len(loaded), "n_permutations": args.permutations, "mean": summary, "pairs": pairs},
                  f, indent=2)
    print(summary)


if __name__ == "__main__":
    main()
