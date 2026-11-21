"""B-cluster on S2 out of sample: cluster both collections, align groups to groups, then match inside groups.

  1. k-means (K groups, on unit-length vectors) of the `oos.n_match` matched training tracks and of the
     gallery paintings, separately.
  2. Group-level Gromov-Wasserstein between the two sets of group centroids (cosine distances), weighted
     by group size, best of `cluster.group_restarts` random restarts; the group coupling is turned into a
     one-to-one pairing of groups by the Hungarian algorithm.
  3. Inside every pair of groups, Gromov-Wasserstein between the member tracks and the member paintings
     gives each matched track a barycentric target, as in `oos.py`.
  4. Test tracks are placed (knn or ridge, from their audio codes) and assigned jointly, one painting each,
     as in `oos.py`.

Controls: `random_groups`, the same steps with a random one-to-one pairing of groups (tests whether the
group-level alignment matters); `shuffled`, a random one-to-one pairing of matched tracks and paintings.
The flat `oos.py` result is the reference.

Usage:
  python methods/s2/cluster.py [--k 8 16 32] [--codes mean standardised] [--gallery runs/gan_pool_<name>/pool.npz]
"""
import time

import numpy as np
import torch
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import full_scores
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, cosine_distances, gromov_wasserstein, load_art, run_name
from match import N_AXES, axes_of
from oos import joint, place_knn, place_linear, unit


def groups_of(x, k, seed):
    model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(unit(x))
    return model.labels_, unit(model.cluster_centers_)


def pair_groups(audio_centres, audio_sizes, art_centres, art_sizes, hp, device, restarts):
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    d_a, d_b = cosine_distances(tensor(audio_centres)), cosine_distances(tensor(art_centres))
    p, q = tensor(audio_sizes / audio_sizes.sum()), tensor(art_sizes / art_sizes.sum())
    best = None
    for r in range(restarts):
        coupling, value = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"],
                                             torch.Generator().manual_seed(r), eps_start=hp["epsilon_start"],
                                             decay=hp["epsilon_decay"], p=p, q=q)
        if best is None or value < best[1]:
            best = (coupling, value)
    rows, cols = linear_sum_assignment(-best[0].cpu().numpy())
    pairing = np.empty(len(rows), dtype=int)
    pairing[rows] = cols
    return pairing, best[1]


def targets_within(matched, audio_labels, gallery, art_labels, pairing, hp, device):
    targets = np.empty((len(matched), gallery.shape[1]), dtype=np.float32)
    values = []
    for a, b in enumerate(pairing):
        rows, cols = np.flatnonzero(audio_labels == a), np.flatnonzero(art_labels == b)
        d_a = cosine_distances(torch.as_tensor(matched[rows], dtype=torch.float32, device=device))
        d_b = cosine_distances(torch.as_tensor(gallery[cols], dtype=torch.float32, device=device))
        coupling, value = gromov_wasserstein(d_a, d_b, hp["epsilon"], hp["outer"], hp["inner"],
                                             torch.Generator().manual_seed(a), eps_start=hp["epsilon_start"],
                                             decay=hp["epsilon_decay"])
        weights = (coupling / coupling.sum(dim=1, keepdim=True)).cpu().numpy()
        targets[rows] = weights @ gallery[cols]
        values.append(value)
    return targets, values


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--codes", nargs="+", default=None)
    parser.add_argument("--k", type=int, nargs="+", default=None)
    parser.add_argument("--n-match", type=int, default=None)
    parser.add_argument("--permutations", type=int, default=None)
    parser.add_argument("--gallery", default=None, help="pool.npz with names and DINOv2 features")
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    hp = {**config["match"], **config["oos"], **config["cluster"]}
    if args.smoke:
        hp.update(outer=20, inner=20, epsilon=0.005, n_match=200, k_values=[4], group_restarts=3)
    hp["n_match"] = args.n_match or hp["n_match"]
    k_values = args.k or hp["k_values"]
    codes_wanted = args.codes or hp["codes"]
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])
    config["run"] = {"codes": codes_wanted, "gallery": args.gallery, "k_values": k_values, **hp}
    log = RunLog(config["paths"]["runs_dir"], run_name("eval_cluster" + ("_pool" if args.gallery else ""), args.smoke),
                 config, seed)

    ids, layers, splits = load_layers(config, args.smoke)
    names, art, art_splits = load_art(config, device, args.smoke)
    gallery, gallery_names = art[art_splits == "train"], names[art_splits == "train"]
    if args.gallery:
        with np.load(args.gallery) as pool:
            gallery, gallery_names = pool["features"].astype(np.float32), pool["names"]
    test_art = art[art_splits == "test"]
    art_mean, art_axes = axes_of(gallery, N_AXES)

    results = {}
    for code in codes_wanted:
        start = time.time()
        codes, _ = codes_of(layers, splits, code, config["audio_code"]["whiten_dim"], config["audio_code"]["layers"])
        tracks = codes.mean(axis=1)
        test_codes, train_codes = tracks[splits == "test"], tracks[splits == "train"]
        matched = train_codes[rng.choice(len(train_codes), size=min(hp["n_match"], len(train_codes)), replace=False)]
        code_mean, code_axes = axes_of(train_codes, N_AXES)
        audio_scores = (test_codes - code_mean) @ code_axes

        def score(picks):
            painting_scores = (gallery[picks] - art_mean) @ art_axes
            return [float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)]

        def evaluate(targets):
            out = {}
            for method in ("knn", "linear"):
                placed = (place_knn(matched, targets, test_codes, hp["k"]) if method == "knn"
                          else place_linear(matched, targets, test_codes, hp["alpha"]))
                picks = joint(placed, gallery)
                out[method] = {**full_scores(test_codes, gallery[picks], test_art, config["eval"]["knn_k"], n_perm, seed),
                               "n_distinct_paintings": int(len(np.unique(picks))), "corr": score(picks)}
                out[method]["picks"] = picks
            return out

        results[code] = {"n_matched": int(len(matched)), "n_test": int(len(test_codes)), "by_k": {}}
        picks_of = {}
        for k in k_values:
            audio_labels, audio_centres = groups_of(matched, k, seed)
            art_labels, art_centres = groups_of(gallery, k, seed)
            audio_sizes = np.bincount(audio_labels, minlength=k).astype(float)
            art_sizes = np.bincount(art_labels, minlength=k).astype(float)
            pairing, group_value = pair_groups(audio_centres, audio_sizes, art_centres, art_sizes, hp, device,
                                               hp["group_restarts"])
            random_pairing = rng.permutation(k)
            block = {"group_gw": group_value, "pairing": pairing.tolist(),
                     "audio_sizes": audio_sizes.astype(int).tolist(), "art_sizes": art_sizes.astype(int).tolist()}
            for name, pair in (("cluster", pairing), ("random_groups", random_pairing)):
                targets, values = targets_within(matched, audio_labels, gallery, art_labels, pair, hp, device)
                scored = evaluate(targets)
                for method, s in scored.items():
                    picks_of[f"k{k}_{name}_{method}"] = s.pop("picks")
                block[name] = {"within_gw_mean": float(np.mean(values)), **scored}
                for method in ("knn", "linear"):
                    s = scored[method]
                    print(f"{code} K {k} {name} {method}: r {s['structure']['mantel_pearson']:.3f} "
                          f"FD {s['image_distribution']['frechet_distance']:.1f} distinct {s['n_distinct_paintings']} "
                          f"corr {np.round(s['corr'], 2)} {time.time() - start:.0f} s", flush=True)
            results[code]["by_k"][str(k)] = block
            log.save_json("results.json", results)
        shuffled = evaluate(gallery[rng.permutation(len(gallery))[np.arange(len(matched)) % len(gallery)]])
        for method, s in shuffled.items():
            s.pop("picks")
            print(f"{code} shuffled {method}: r {s['structure']['mantel_pearson']:.3f}", flush=True)
        results[code]["shuffled"] = shuffled
        results[code]["seconds"] = round(time.time() - start, 1)
        np.savez(log.dir / f"joint_{code}.npz", track_ids=ids[splits == "test"], methods=np.array(list(picks_of)),
                 painting_names=gallery_names[np.stack(list(picks_of.values()))])
        log.save_json("results.json", results)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
