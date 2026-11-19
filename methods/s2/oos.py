"""S2 out of sample: match training tracks to the training paintings by shape, then place unseen test tracks.

`match.py` solves the match on the test tracks themselves (transductive). Here the Gromov-Wasserstein match
uses only `oos.n_match` random training tracks; the test tracks take no part in it. Each matched track gets
a barycentric target, the coupling-weighted mean of the painting features it is coupled with. A test track
is then placed from its audio code alone:

  knn      mean of the targets of its `oos.k` nearest matched tracks (cosine distance of the audio codes)
  linear   ridge regression from audio code to target, fitted on the matched tracks

and receives the training painting nearest (cosine) to its placed feature. Scored like the painting block
of F1. Control `shuffled`: the same placements from a random one-to-one pairing of the matched tracks with
paintings instead of the GW coupling.

Per restart (`seeds` rows, format read by `methods/s4/d_s4.py`): orientation `corr` of the knn placement
on the leading principal axes, GW objective, distinct paintings. `assignments_<code>.npz` holds the knn
assignment of every restart.

Usage:
  python methods/s2/oos.py [--codes mean standardised] [--restarts 3] [--n-match 3000]
"""
import time

import numpy as np
import torch

from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import full_scores
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, cosine_distances, gromov_wasserstein, load_art, run_name
from match import N_AXES, axes_of


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def place_knn(train_codes, targets, test_codes, k):
    sims = unit(test_codes) @ unit(train_codes).T
    neighbours = np.argsort(-sims, axis=1)[:, :k]
    return targets[neighbours].mean(axis=1)


def place_linear(train_codes, targets, test_codes, alpha):
    mean_x, mean_y = train_codes.mean(0), targets.mean(0)
    x = train_codes - mean_x
    scale = alpha * np.trace(x.T @ x) / x.shape[1]
    weights = np.linalg.solve(x.T @ x + scale * np.eye(x.shape[1]), x.T @ (targets - mean_y))
    return (test_codes - mean_x) @ weights + mean_y


def nearest(placed, gallery):
    return np.argmax(unit(placed) @ unit(gallery).T, axis=1)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--restarts", type=int, default=None)
    parser.add_argument("--n-match", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    hp = {**config["match"], **config["oos"]}
    if args.smoke:
        hp.update(outer=20, inner=20, restarts=2, epsilon=0.005, n_match=200)
    hp["restarts"] = args.restarts or hp["restarts"]
    hp["n_match"] = args.n_match or hp["n_match"]
    codes_wanted = args.codes or hp["codes"]
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])
    config["run"] = {"codes": codes_wanted, **hp}
    log = RunLog(config["paths"]["runs_dir"], run_name("eval_oos", args.smoke), config, seed)

    ids, layers, splits = load_layers(config, args.smoke)
    names, art, art_splits = load_art(config, device, args.smoke)
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    test_art = art[art_splits == "test"]
    art_mean, art_axes = axes_of(gallery, N_AXES)
    d_b = cosine_distances(torch.as_tensor(gallery, dtype=torch.float32, device=device))

    results = {}
    for code in codes_wanted:
        start = time.time()
        codes, _ = codes_of(layers, splits, code, config["audio_code"]["whiten_dim"], config["audio_code"]["layers"])
        tracks = codes.mean(axis=1)
        test_codes, train_codes = tracks[splits == "test"], tracks[splits == "train"]
        matched = train_codes[rng.choice(len(train_codes), size=min(hp["n_match"], len(train_codes)), replace=False)]
        d_a = cosine_distances(torch.as_tensor(matched, dtype=torch.float32, device=device))
        code_mean, code_axes = axes_of(train_codes, N_AXES)
        audio_scores = (test_codes - code_mean) @ code_axes

        def score(picks):
            painting_scores = (gallery[picks] - art_mean) @ art_axes
            return [float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)]

        rows, targets_of, knn_picks = [], [], []
        for r in range(hp["restarts"]):
            generator = torch.Generator().manual_seed(r)
            coupling, value = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"], generator,
                                                 eps_start=hp["epsilon_start"], decay=hp["epsilon_decay"])
            weights = (coupling / coupling.sum(dim=1, keepdim=True)).cpu().numpy()
            targets = weights @ gallery
            picks = nearest(place_knn(matched, targets, test_codes, hp["k"]), gallery)
            targets_of.append(targets)
            knn_picks.append(picks)
            rows.append({"seed": r, "gw": value, "corr": score(picks), "distinct": int(len(np.unique(picks))),
                         "row_concentration": float((coupling.max(dim=1).values / coupling.sum(dim=1)).mean())})
            print(f"{code} restart {r}: GW {value:.4f}, distinct {rows[-1]['distinct']}, corr",
                  np.round(rows[-1]["corr"], 3), f"{time.time() - start:.0f} s", flush=True)
        best = int(np.argmin([row["gw"] for row in rows]))
        shuffled_targets = gallery[rng.permutation(len(gallery))[np.arange(len(matched)) % len(gallery)]]

        placements = {}
        for source, targets in (("gw", targets_of[best]), ("shuffled", shuffled_targets)):
            for method in ("knn", "linear"):
                placed = (place_knn(matched, targets, test_codes, hp["k"]) if method == "knn"
                          else place_linear(matched, targets, test_codes, hp["alpha"]))
                picks = nearest(placed, gallery)
                _, counts = np.unique(picks, return_counts=True)
                placements[f"{source}_{method}"] = {
                    **full_scores(test_codes, gallery[picks], test_art, config["eval"]["knn_k"], n_perm, seed),
                    "n_distinct_paintings": int(len(counts)), "largest_share": float(counts.max() / len(picks)),
                    "corr": score(picks)}
                block = placements[f"{source}_{method}"]
                print(f"  {code} {source}_{method}: r {block['structure']['mantel_pearson']:.3f}",
                      f"FD {block['image_distribution']['frechet_distance']:.1f} distinct {len(counts)}", flush=True)

        signs = np.sign(np.array([row["corr"] for row in rows]))
        same = [float((knn_picks[a] == knn_picks[b]).mean())
                for a in range(len(knn_picks)) for b in range(a + 1, len(knn_picks))]
        results[code] = {
            "n_matched": int(len(matched)), "n_test": int(len(test_codes)), "n_paintings": int(len(gallery)),
            "gw": {"selected_restart": best, "values": [row["gw"] for row in rows]},
            "placements": placements,
            "orientation_counts": {f"axis{k + 1}_positive": int((signs[:, k] > 0).sum()) for k in range(N_AXES)},
            "between_restarts": {"same_painting_share": float(np.mean(same)) if same else None},
            "seeds": rows,
            "seconds": round(time.time() - start, 1)}
        np.savez(log.dir / f"assignments_{code}.npz", track_ids=ids[splits == "test"],
                 seeds=np.arange(hp["restarts"]), painting_names=gallery_names[np.stack(knn_picks)])
        log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
