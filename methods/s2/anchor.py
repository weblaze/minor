"""B-anchor on S2 out of sample: the orientation of the shape match is chosen by a machine judge.

Shape matching cannot tell a mapping from its mirror image along a principal axis. Here every orientation
of the three leading axes is tried and CAV-MAE (S4, trained on natural audio-visual co-occurrence) picks
one, without labels and without human input.

  1. For each sign pattern s of the leading `N_AXES` principal axes (8 patterns), Gromov-Wasserstein
     between the `oos.n_match` matched training tracks and the gallery paintings is started from the
     transport plan of the cost -sum_k s_k a_ik b_jk (a, b: standardised axis scores), so that it starts in
     that orientation, at a starting regularisation `--eps-start` low enough that the start is not blurred
     away (with the configured 0.05 every start ended in the same coupling, run log 2026-11-21). With
     `--alpha` > 0 the oriented cost also stays in the objective (fused GW, weight alpha), so the match is
     held in that orientation. The orientation it ends in is measured on the training tracks (`train_corr`).
  2. Judge, on the training tracks only: the mean CAV-MAE similarity of each matched track and its row-maximum
     painting, as z against random re-assignments of the same paintings (as `methods/s4/d_s4.py`).
  3. The candidate with the highest judge z is the anchored match; the one with the lowest GW objective is
     the unanchored reference. Every candidate's test tracks are placed (ridge) and assigned jointly as in
     `oos.py`, and scored: Mantel r, FD, orientation, and the judge z on the held-out test pairs, which shows
     whether the anchor chosen on training tracks carries over.

Usage:
  python methods/s2/anchor.py [--codes mean standardised] [--eps-start 0.005] [--alpha 0 0.1]
"""
import itertools
from pathlib import Path
import time

import numpy as np
import torch

from abstraction.data.embeddings import load_shards
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import full_scores
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, cosine_distances, gromov_wasserstein, load_art, run_name, sinkhorn_log
from match import N_AXES, axes_of
from oos import joint, place_linear, unit


def judge_z(s, tracks, paintings, n_null, rng):
    observed = float(s[tracks, paintings].mean())
    null = np.array([s[tracks, rng.permutation(paintings)].mean() for _ in range(n_null)])
    return float((observed - null.mean()) / null.std())


def oriented_start(a_scores, b_scores, signs, device, iterations):
    """Oriented linear cost (standardised, minimum 0) and its entropic transport plan."""
    a = a_scores / a_scores.std(axis=0)
    b = b_scores / b_scores.std(axis=0)
    cost = -(a * signs) @ b.T
    cost = (cost - cost.min()) / cost.std()
    cost = torch.as_tensor(cost, dtype=torch.float32, device=device)
    n, m = cost.shape
    log_p = torch.full((n,), -np.log(n), device=device)
    log_q = torch.full((m,), -np.log(m), device=device)
    return cost, sinkhorn_log(cost, log_p, log_q, 0.1, iterations)


def gw_objective(d_a, d_b, coupling):
    n, m = coupling.shape
    constant = (d_a ** 2).mean(dim=1)[:, None] + (d_b ** 2).mean(dim=1)[None, :]
    return float(((constant - 2 * d_a @ coupling @ d_b) * coupling).sum())


def corr_of(x, y):
    return [float(np.corrcoef(x[:, k], y[:, k])[0, 1]) for k in range(x.shape[1])]


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--n-match", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--embeddings", default=str(Path(__file__).resolve().parents[2] / "datasets" / "cav_mae_embeddings"))
    parser.add_argument("--null", type=int, default=999)
    parser.add_argument("--eps-start", type=float, default=0.005)
    parser.add_argument("--alpha", type=float, nargs="+", default=[0.0, 0.1])
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    hp = {**config["match"], **config["oos"]}
    if args.smoke:
        hp.update(outer=20, inner=20, epsilon=0.005, n_match=200)
    hp["n_match"] = args.n_match or hp["n_match"]
    codes_wanted = args.codes or hp["codes"]
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])
    config["run"] = {"codes": codes_wanted, "embeddings": args.embeddings, "eps_start": args.eps_start,
                     "alpha": args.alpha, **hp}
    log = RunLog(config["paths"]["runs_dir"], run_name("eval_anchor", args.smoke), config, seed)

    ids, layers, splits = load_layers(config, args.smoke)
    names, art, art_splits = load_art(config, device, args.smoke)
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    test_art = art[art_splits == "test"]
    art_mean, art_axes = axes_of(gallery, N_AXES)
    d_b = cosine_distances(torch.as_tensor(gallery, dtype=torch.float32, device=device))

    if args.smoke:
        cav_ids, cav_names = ids, gallery_names
        cav_audio = rng.normal(size=(len(ids), 1, 16))
        cav_images = rng.normal(size=(len(gallery_names), 16))
    else:
        cav_ids, cav_audio = load_shards(f"{args.embeddings}/audio")
        with np.load(f"{args.embeddings}/art.npz") as data:
            cav_names, cav_images = data["names"], data["features"]
    track_row = {str(t): i for i, t in enumerate(cav_ids)}
    name_row = {str(n): i for i, n in enumerate(cav_names)}
    cav_tracks = unit(np.asarray(cav_audio, dtype=np.float32).mean(axis=1))
    cav_gallery = unit(np.asarray(cav_images, dtype=np.float32)[[name_row[str(n)] for n in gallery_names]])

    def judge(track_ids, picks):
        keep = np.array([str(t) in track_row for t in track_ids])
        rows = np.array([track_row[str(t)] for t in track_ids[keep]])
        s = cav_tracks[rows] @ cav_gallery.T
        return judge_z(s, np.arange(len(rows)), picks[keep], args.null, rng)

    results = {}
    for code in codes_wanted:
        start = time.time()
        codes, _ = codes_of(layers, splits, code, config["audio_code"]["whiten_dim"], config["audio_code"]["layers"])
        tracks = codes.mean(axis=1)
        train_ids, test_ids = ids[splits == "train"], ids[splits == "test"]
        test_codes, train_codes = tracks[splits == "test"], tracks[splits == "train"]
        chosen = rng.choice(len(train_codes), size=min(hp["n_match"], len(train_codes)), replace=False)
        matched, matched_ids = train_codes[chosen], train_ids[chosen]
        code_mean, code_axes = axes_of(train_codes, N_AXES)
        a_scores = (matched - code_mean) @ code_axes
        b_scores = (gallery - art_mean) @ art_axes
        test_scores = (test_codes - code_mean) @ code_axes
        d_a = cosine_distances(torch.as_tensor(matched, dtype=torch.float32, device=device))

        candidates = []
        for alpha, signs in itertools.product(args.alpha, itertools.product((1, -1), repeat=N_AXES)):
            cost, init = oriented_start(a_scores, b_scores, np.array(signs), device, hp["inner"])
            coupling, _ = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"],
                                             eps_start=args.eps_start, decay=hp["epsilon_decay"], init=init,
                                             feature_cost=cost, alpha=alpha)
            value = gw_objective(d_a, d_b, coupling)
            weights = (coupling / coupling.sum(dim=1, keepdim=True)).cpu().numpy()
            targets = weights @ gallery
            row_max = coupling.argmax(dim=1).cpu().numpy()
            placed = place_linear(matched, targets, test_codes, hp["alpha"])
            picks = joint(placed, gallery)
            block = {"alpha": alpha, "start_signs": list(signs), "gw": value,
                     "train_corr": corr_of(a_scores, (targets - art_mean) @ art_axes),
                     "train_judge_z": judge(matched_ids, row_max),
                     "test": {**full_scores(test_codes, gallery[picks], test_art, config["eval"]["knn_k"], n_perm, seed),
                              "n_distinct_paintings": int(len(np.unique(picks))),
                              "corr": corr_of(test_scores, (gallery[picks] - art_mean) @ art_axes),
                              "judge_z": judge(test_ids, picks)}}
            candidates.append(block)
            t = block["test"]
            print(f"{code} alpha {alpha} start {signs}: GW {value:.4f} train corr {np.round(block['train_corr'], 2)} "
                  f"train z {block['train_judge_z']:.2f} | test r {t['structure']['mantel_pearson']:.3f} "
                  f"FD {t['image_distribution']['frechet_distance']:.1f} z {t['judge_z']:.2f} "
                  f"{time.time() - start:.0f} s", flush=True)
        results[code] = {"n_matched": int(len(matched)), "n_test": int(len(test_codes)),
                         "candidates": candidates, "seconds": round(time.time() - start, 1)}
        for alpha in args.alpha:
            index = [i for i, c in enumerate(candidates) if c["alpha"] == alpha]
            train_z = np.array([candidates[i]["train_judge_z"] for i in index])
            test_z = np.array([candidates[i]["test"]["judge_z"] for i in index])
            anchored = index[int(np.argmax(train_z))]
            lowest = index[int(np.argmin([candidates[i]["gw"] for i in index]))]
            results[code][f"alpha_{alpha}"] = {"anchored": anchored, "lowest_gw": lowest,
                                               "train_test_z_corr": float(np.corrcoef(train_z, test_z)[0, 1])}
            print(f"{code} alpha {alpha}: anchored candidate {anchored} {candidates[anchored]['start_signs']}, "
                  f"lowest GW {lowest} {candidates[lowest]['start_signs']}, "
                  f"train/test judge z corr {results[code][f'alpha_{alpha}']['train_test_z_corr']:.2f}", flush=True)
        log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
