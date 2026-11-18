"""S2: blind matching of the test tracks to the training paintings by the shape of the two spaces.

Nothing is trained and no pair is used. The cosine distance matrix of the test tracks' audio codes and
that of the training paintings' DINOv2 features (each divided by its mean) are matched by entropic
Gromov-Wasserstein (GW): a coupling of tracks and paintings under which pairs of tracks have the same
distance as the pairs of paintings they are coupled with. GW is not convex; `match.restarts` random
initial couplings are solved and the one with the lowest objective is the match (selected without labels).

Reported per audio code:

  gw            objective of each restart, and of the selected match
  reference     GW of the same tracks against (a) `gaussian`: samples of a Gaussian with the mean and
                covariance of the painting features (same shape statistics up to second order, no further
                structure), and (b) `audio`: random sets of training tracks of the same size (the same
                space; the value a perfect shape match would approach). Each reference is solved with
                `null_restarts` restarts, and compared with the best of the first `null_restarts`
                restarts of the real match
  painting      each test track receives the painting with the largest coupling weight; scored as the
                painting block of F1 (audio codes against DINOv2 features of the assigned paintings)
  seeds         per restart: `corr`, the correlation of the test tracks' scores on principal axis k of
                the audio code with the scores of their paintings on principal axis k of the art features
                (the orientation of axis k), objective, distinct paintings
  between_restarts  mean cosine similarity of the paintings two restarts give the same track, and the
                share of tracks that two restarts give the same painting

`assignments_<code>.npz` holds the assignment of every restart in the format read by
`methods/s4/d_s4.py --runs <this run>`; `features_<code>.npz` the selected match in the format read by
`methods/s4/probe.py --assignments`.

Usage:
  python methods/s2/match.py [--codes mean standardised] [--restarts 20]
"""
import time

import numpy as np
import torch

from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import full_scores
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, cosine_distances, gromov_wasserstein, load_art, run_name

N_AXES = 3


def axes_of(x, k):
    """Leading principal axes, with the sign convention of methods/f1/orientation.py (eigh of the covariance),
    so that orientations can be compared between S2 and F1."""
    mean = x.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(eigval)[::-1][:k]
    return mean, eigvec[:, order]


def solve(d_a, target, hp, seeds, device, concentration=None):
    """GW objective of each restart with the given seeds, and the assignment (row maximum) of each.

    `concentration`, if a list, receives per restart the mean share of a row's mass on its largest entry."""
    d_b = cosine_distances(torch.as_tensor(target, dtype=torch.float32, device=device))
    values, couplings = [], []
    concentration = [] if concentration is None else concentration
    for seed in seeds:
        generator = torch.Generator().manual_seed(int(seed))
        coupling, value = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"], generator,
                                             eps_start=hp["epsilon_start"], decay=hp["epsilon_decay"])
        values.append(value)
        couplings.append(coupling.argmax(dim=1).cpu().numpy())
        concentration.append(float((coupling.max(dim=1).values / coupling.sum(dim=1)).mean()))
    return values, couplings


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    hp = dict(config["match"])
    if args.smoke:
        hp.update(outer=20, inner=20, restarts=3, n_null=2, null_restarts=2, epsilon=0.005)
    hp["restarts"] = args.restarts or hp["restarts"]
    codes_wanted = args.codes or hp["codes"]
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])
    config["run"] = {"codes": codes_wanted, **hp}
    log = RunLog(config["paths"]["runs_dir"], run_name("eval_match", args.smoke), config, seed)

    ids, layers, splits = load_layers(config, args.smoke)
    names, art, art_splits = load_art(config, device, args.smoke)
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    test_art = art[art_splits == "test"]
    art_mean, art_axes = axes_of(gallery, N_AXES)
    gaussian_cov = np.cov(gallery, rowvar=False)

    results = {}
    for code in codes_wanted:
        start = time.time()
        codes, _ = codes_of(layers, splits, code, config["audio_code"]["whiten_dim"], config["audio_code"]["layers"])
        tracks = codes.mean(axis=1)
        test_codes, train_codes = tracks[splits == "test"], tracks[splits == "train"]
        d_a = cosine_distances(torch.as_tensor(test_codes, dtype=torch.float32, device=device))
        code_mean, code_axes = axes_of(train_codes, N_AXES)
        audio_scores = (test_codes - code_mean) @ code_axes

        restart_seeds = np.arange(hp["restarts"])
        concentration = []
        values, picks = solve(d_a, gallery, hp, restart_seeds, device, concentration)
        best = int(np.argmin(values))
        rows = []
        for r, nearest in zip(restart_seeds, picks):
            painting_scores = (gallery[nearest] - art_mean) @ art_axes
            corr = [float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)]
            rows.append({"seed": int(r), "corr": corr, "gw": values[r], "distinct": int(len(np.unique(nearest))),
                         "row_concentration": concentration[r]})

        n_ref = hp["null_restarts"]
        reference = {"gaussian": [], "audio": []}
        for draw in range(hp["n_null"]):
            sample = rng.multivariate_normal(gallery.mean(axis=0), gaussian_cov, size=len(gallery))
            reference["gaussian"].append(min(solve(d_a, sample, hp, 1000 + np.arange(n_ref), device)[0]))
            subset = train_codes[rng.choice(len(train_codes), size=min(len(gallery), len(train_codes)), replace=False)]
            reference["audio"].append(min(solve(d_a, subset, hp, 2000 + np.arange(n_ref), device)[0]))
        real_comparable = min(values[:n_ref])
        reference_summary = {}
        for key, draws in reference.items():
            draws = np.array(draws)
            reference_summary[key] = {"values": draws.tolist(), "mean": float(draws.mean()), "std": float(draws.std()),
                                      "z_real": float((real_comparable - draws.mean()) / (draws.std() + 1e-12))}

        nearest = picks[best]
        unit = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
        between = [float((unit[picks[a]] * unit[picks[b]]).sum(1).mean())
                   for a in range(len(picks)) for b in range(a + 1, len(picks))]
        _, counts = np.unique(nearest, return_counts=True)
        signs = np.sign(np.array([row["corr"] for row in rows]))
        same = [float((picks[a] == picks[b]).mean()) for a in range(len(picks)) for b in range(a + 1, len(picks))]
        results[code] = {
            "n_tracks": int(len(test_codes)), "n_paintings": int(len(gallery)),
            "gw": {"selected_restart": best, "selected": values[best], "values": values,
                   "best_of_first_null_restarts": real_comparable},
            "reference": reference_summary,
            "painting": {**full_scores(test_codes, gallery[nearest], test_art, config["eval"]["knn_k"], n_perm, seed),
                         "n_distinct_paintings": int(len(counts)), "largest_share": float(counts.max() / len(nearest)),
                         "corr": rows[best]["corr"]},
            "orientation_counts": {f"axis{k + 1}_positive": int((signs[:, k] > 0).sum()) for k in range(N_AXES)},
            "between_restarts": {"painting_similarity": float(np.mean(between)) if between else None,
                                 "same_painting_share": float(np.mean(same)) if same else None,
                                 "painting_similarity_shuffled": float((unit[picks[0]] * unit[rng.permutation(picks[0])]).sum(1).mean())},
            "seeds": rows,
            "seconds": round(time.time() - start, 1)}
        np.savez(log.dir / f"assignments_{code}.npz", track_ids=ids[splits == "test"], seeds=restart_seeds,
                 painting_names=gallery_names[np.stack(picks)])
        np.savez(log.dir / f"features_{code}.npz", track_ids=ids[splits == "test"], audio_codes=test_codes,
                 painting_features=gallery[nearest], painting_names=gallery_names[nearest])
        log.save_json("results.json", results)
        block = results[code]
        print(f"== {code}: GW selected {block['gw']['selected']:.4f}; references",
              {k: round(v["mean"], 4) for k, v in reference_summary.items()},
              "| painting r", round(block["painting"]["structure"]["mantel_pearson"], 3),
              "FD", round(block["painting"]["image_distribution"]["frechet_distance"], 1),
              "distinct", block["painting"]["n_distinct_paintings"], "|", block["orientation_counts"],
              f"| {block['seconds']} s")
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
