"""D-S4: CAV-MAE scores the orientations of a set of mappings.

Input: the `assignments_<config>.npz` files written by `methods/f1/orientation.py --save-assignments`
(test track ids, and the painting assigned to each track by each seed) and the `results.json` of the
same run (per seed, `corr`: the correlation between the audio and painting scores on each principal
axis; its sign is the orientation of that axis).

For each seed the score is the mean CAV-MAE similarity S of the assigned track-painting pairs,
expressed as z against random re-assignments of the same paintings to the same tracks. The
re-assignment keeps every track and every painting, so only the pair part of S enters the score.

For each principal axis k the seeds are split by the sign of corr_k. The script reports the mean z of
each group, the difference, and a permutation p-value of the difference over the seed labels; and
the Spearman correlation between corr_k and z over the seeds. If CAV-MAE prefers one orientation of
axis k, the seeds of that orientation have the higher z.

Usage:
  python methods/s4/d_s4.py --runs runs/f1_orientation_dS4_* [--embeddings DIR]
"""
import glob
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from abstraction.pipeline.cli import setup
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, load_cav_embeddings, run_name, unit
from probe import assignment_score

N_LABEL_PERMUTATIONS = 9999


def sign_split(z, corr, rng):
    positive = corr > 0
    if positive.all() or not positive.any():
        return {"n_positive": int(positive.sum()), "n_negative": int((~positive).sum())}
    difference = z[positive].mean() - z[~positive].mean()
    null = np.empty(N_LABEL_PERMUTATIONS)
    for i in range(N_LABEL_PERMUTATIONS):
        labels = rng.permutation(positive)
        null[i] = z[labels].mean() - z[~labels].mean()
    rho, rho_p = spearmanr(corr, z)
    return {"n_positive": int(positive.sum()), "n_negative": int((~positive).sum()),
            "z_positive": float(z[positive].mean()), "z_negative": float(z[~positive].mean()),
            "difference": float(difference),
            "p_two_sided": float((1 + (np.abs(null) >= abs(difference)).sum()) / (1 + N_LABEL_PERMUTATIONS)),
            "spearman_corr_z": float(rho), "spearman_p": float(rho_p)}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--runs", nargs="+", required=True, help="orientation run directories (globs allowed)")
    parser.add_argument("--null", type=int, default=999, help="random re-assignments per seed")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    embeddings = args.embeddings or config["cav_mae"]["embeddings"]
    runs = sorted({path for pattern in args.runs for path in glob.glob(pattern)})
    config["run"] = {"embeddings": embeddings, "runs": runs}
    log = RunLog(config["paths"]["runs_dir"], run_name("d_s4", args.smoke), config, seed)

    cav_ids, cav_audio, cav_names, cav_images = load_cav_embeddings(embeddings)
    track_position = {str(t): i for i, t in enumerate(cav_ids)}
    name_position = {str(n): i for i, n in enumerate(cav_names)}
    audio, images = unit(cav_audio.mean(axis=1)), unit(cav_images)

    results = {}
    for run in runs:
        with open(Path(run) / "results.json") as f:
            orientation = json.load(f)
        for path in sorted(Path(run).glob("assignments_*.npz")):
            name = path.stem.removeprefix("assignments_")
            with np.load(path) as data:
                track_ids, seeds, painting_names = data["track_ids"], data["seeds"], data["painting_names"]
            keep = np.array([str(t) in track_position for t in track_ids])
            tracks = np.array([track_position[str(t)] for t in track_ids[keep]])
            s = audio[tracks] @ images.T
            rows = {row["seed"]: row for row in orientation[name]["seeds"]}
            per_seed = []
            for seed_value, names in zip(seeds, painting_names):
                paintings = np.array([name_position[str(n)] for n in names[keep]])
                score = assignment_score(s, np.arange(len(tracks)), paintings, args.null, rng)
                per_seed.append({"seed": int(seed_value), "corr": rows[int(seed_value)]["corr"], **score})
            z = np.array([row["z"] for row in per_seed])
            corr = np.array([row["corr"] for row in per_seed])
            results[f"{Path(run).name}/{name}"] = {
                "n_seeds": len(per_seed), "n_tracks": int(len(tracks)),
                "z_mean": float(z.mean()), "z_std": float(z.std()),
                "n_seeds_p_below_0.05": int(sum(row["p_greater"] < 0.05 for row in per_seed)),
                "axes": [sign_split(z, corr[:, k], rng) for k in range(corr.shape[1])],
                "seeds": per_seed}
            log.save_json("results.json", results)

    log.save_json("results.json", results)
    for key, block in results.items():
        print(f"== {key}: {block['n_seeds']} seeds, z {block['z_mean']:.2f} (sd {block['z_std']:.2f}), "
              f"{block['n_seeds_p_below_0.05']} seeds with p < 0.05")
        for k, axis in enumerate(block["axes"]):
            print(f"   axis {k + 1}:", {key: round(v, 3) if isinstance(v, float) else v for key, v in axis.items()})
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
