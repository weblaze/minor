"""B-anchor and B-cluster on F1.

One F1 mapping is trained per seed (`fit` of realism.py, config `--config`, default `swd` as in the D-vote
and D-S4 runs). For every seed:

B-anchor   CAV-MAE (S4) judges the seed on the validation tracks: mean CAV-MAE similarity of each track and
           its nearest painting, as z against random re-assignments of the same paintings. The anchored
           mapping is the seed with the highest validation z; it is compared on the test tracks with every
           other seed, with the seed of lowest validation SWD, and with the orientation of the seed majority
           (D-vote). The correlation of validation z and test z over the seeds shows whether the judge's
           choice carries over to held-out tracks.

B-cluster  k-means groups of the training tracks (mean code) and of the gallery paintings; the groups are
           paired by Gromov-Wasserstein between the group centroids weighted by group size (the group step
           of methods/s2/cluster.py, best of `group_restarts`), made one-to-one by the Hungarian algorithm.
           A test track is assigned to its nearest track group and receives the painting nearest to its F1
           output among the paintings of the paired group. Control: a random one-to-one pairing of groups.

Test scores per seed: Mantel r between audio distances and painting distances, orientation corr on the leading
axes, distinct paintings, CAV-MAE judge z.

Usage:
  python methods/f1/b_tasks.py --code mean [--seeds 20] [--k 8 16] [--config swd]
"""
import importlib.util
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans

from abstraction.data.embeddings import load_shards
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.metrics.structure import distance_matrix
from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import correlation
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, load_art, run_name
from orientation import EXTRA, N_AXES, axes_of
from realism import DEFAULT, GRID, VIEWS, fit, sweep_grid

_spec = importlib.util.spec_from_file_location("s2_common", Path(__file__).parents[1] / "s2" / "common.py")
s2 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s2)

GW = {"epsilon": 0.0005, "epsilon_start": 0.05, "epsilon_decay": 0.95, "outer": 300, "inner": 200}


def unit(x):
    return x / np.linalg.norm(x, axis=1, keepdims=True)


def judge_z(s, picks, n_null, rng):
    rows = np.arange(len(picks))
    observed = float(s[rows, picks].mean())
    null = np.array([s[rows, rng.permutation(picks)].mean() for _ in range(n_null)])
    return float((observed - null.mean()) / null.std())


def pair_groups(audio, art, k, seed, restarts, device):
    audio_model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(unit(audio))
    art_model = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(unit(art))
    tensor = lambda x: torch.as_tensor(x, dtype=torch.float64, device=device)
    d_a = s2.cosine_distances(tensor(unit(audio_model.cluster_centers_)))
    d_b = s2.cosine_distances(tensor(unit(art_model.cluster_centers_)))
    p = tensor(np.bincount(audio_model.labels_, minlength=k) / len(audio))
    q = tensor(np.bincount(art_model.labels_, minlength=k) / len(art))
    best = None
    for r in range(restarts):
        coupling, value = s2.gromov_wasserstein(d_a, d_b, GW["epsilon"], GW["outer"], GW["inner"],
                                                torch.Generator().manual_seed(r), eps_start=GW["epsilon_start"],
                                                decay=GW["epsilon_decay"], p=p, q=q)
        if best is None or value < best[1]:
            best = (coupling, value)
    rows, cols = linear_sum_assignment(-best[0].cpu().numpy())
    pairing = np.empty(k, dtype=int)
    pairing[rows] = cols
    return audio_model, art_model.labels_, pairing, best[1]


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--config_name", "--config-name", dest="config_name", default="swd")
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--k", type=int, nargs="+", default=[8, 16])
    parser.add_argument("--group-restarts", type=int, default=50)
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--embeddings", default="datasets/cav_mae_embeddings")
    parser.add_argument("--null", type=int, default=999)
    args = parser.parse_args()
    config, device = setup(args)
    steps = args.steps or (40 if args.smoke else config["mapper"]["steps"])
    if args.smoke:
        args.seeds, args.k, args.group_restarts, args.null = 3, [4], 3, 99
        GW.update(outer=20, inner=20, epsilon=0.005)
    set_seed(0)
    rng = np.random.default_rng(0)
    config["run"] = {"config": args.config_name, "seeds": args.seeds, "k": args.k, "steps": steps,
                     "group_restarts": args.group_restarts, "embeddings": args.embeddings}
    log = RunLog(config["paths"]["runs_dir"], run_name("b_tasks", args.smoke), config, 0)

    track_ids, layers, splits = load_layers(config, args.smoke)
    art_names, art, art_splits = load_art(config, device, args.smoke)
    gallery_np, val_art = art[art_splits == "train"], art[art_splits == "val"]
    gallery_names = art_names[art_splits == "train"]
    gallery = torch.from_numpy(gallery_np).to(device)
    code = config["audio_code"]
    codes, _ = codes_of(layers, splits, code["variant"], code["whiten_dim"], code["layers"])
    to = lambda x: torch.from_numpy(x).to(device)
    train = to(codes[splits == "train"])
    train_views = [to(codes_of(layers, splits, "standardised", code["whiten_dim"], [i])[0][splits == "train"])
                   for i in VIEWS]
    train_mean = codes[splits == "train"].mean(axis=1)
    val, test = to(codes[splits == "val"].mean(axis=1)), to(codes[splits == "test"].mean(axis=1))
    test_np = test.cpu().numpy()
    d_audio = distance_matrix(test_np)
    code_mean, code_axes = axes_of(train_mean, N_AXES)
    art_mean, art_axes = axes_of(gallery_np, N_AXES)
    audio_scores = (test_np - code_mean) @ code_axes
    scale = float(gallery_np.std())

    if args.smoke:
        cav_rows = {str(t): i for i, t in enumerate(track_ids)}
        cav_tracks = unit(rng.normal(size=(len(track_ids), 16)))
        cav_gallery = unit(rng.normal(size=(len(gallery_names), 16)))
    else:
        cav_ids, cav_audio = load_shards(f"{args.embeddings}/audio")
        with np.load(f"{args.embeddings}/art.npz") as data:
            name_row = {str(n): i for i, n in enumerate(data["names"])}
            cav_gallery = unit(data["features"].astype(np.float32)[[name_row[str(n)] for n in gallery_names]])
        cav_rows = {str(t): i for i, t in enumerate(cav_ids)}
        cav_tracks = unit(np.asarray(cav_audio, dtype=np.float32).mean(axis=1))

    def judge_of(ids):
        keep = np.array([str(t) in cav_rows for t in ids])
        return keep, cav_tracks[[cav_rows[str(t)] for t in ids[keep]]] @ cav_gallery.T

    val_keep, val_s = judge_of(track_ids[splits == "val"])
    test_keep, test_s = judge_of(track_ids[splits == "test"])

    def scores(picks):
        painting_scores = (gallery_np[picks] - art_mean) @ art_axes
        return {"r_painting": correlation(d_audio, distance_matrix(gallery_np[picks])),
                "corr": [float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)],
                "distinct": int(len(np.unique(picks))), "judge_z": judge_z(test_s, picks[test_keep], args.null, rng)}

    groups = {}
    for k in args.k:
        audio_model, art_labels, pairing, value = pair_groups(train_mean, gallery_np, k, 0, args.group_restarts, device)
        groups[k] = {"test_groups": audio_model.predict(unit(test_np)), "art_labels": art_labels,
                     "pairings": {"gw": pairing, "random_groups": rng.permutation(k)}, "group_gw": value}
        print(f"K {k}: group GW {value:.4f}, pairing {pairing.tolist()}", flush=True)

    unit_gallery = F.normalize(gallery, dim=1)
    spec = {**DEFAULT, **{**GRID, **EXTRA, **sweep_grid()}[args.config_name]}
    rows = []
    for seed in range(args.seeds):
        forward = fit(spec, config, train, train_views, gallery, device, steps, seed)
        with torch.no_grad():
            _, query = forward(test)
            sims = (F.normalize(query, dim=1) @ unit_gallery.T).cpu().numpy()
            _, val_query = forward(val)
            val_picks = (F.normalize(val_query, dim=1) @ unit_gallery.T).argmax(1).cpu().numpy()
            out_val, _ = forward(val)
            torch.manual_seed(0)
            val_swd = float(sliced_wasserstein(out_val / scale, to(val_art) / scale, 2048))
        row = {"seed": seed, "val_judge_z": judge_z(val_s, val_picks[val_keep], args.null, rng), "val_swd": val_swd,
               "test": scores(sims.argmax(1)), "cluster": {}}
        for k, g in groups.items():
            for name, pairing in g["pairings"].items():
                allowed = g["art_labels"][None, :] == pairing[g["test_groups"]][:, None]
                row["cluster"][f"k{k}_{name}"] = scores(np.where(allowed, sims, -np.inf).argmax(1))
        rows.append(row)
        t = row["test"]
        print(f"seed {seed}: val z {row['val_judge_z']:.2f} | test r {t['r_painting']:.3f} corr {np.round(t['corr'], 2)} "
              f"z {t['judge_z']:.2f} distinct {t['distinct']} | "
              + " ".join(f"{key} r {v['r_painting']:.3f}" for key, v in row["cluster"].items()), flush=True)
        log.save_json("results.json", {"seeds": rows})

    val_z = np.array([r["val_judge_z"] for r in rows])
    test_z = np.array([r["test"]["judge_z"] for r in rows])
    axis1 = np.sign([r["test"]["corr"][0] for r in rows])
    majority = float(np.sign(axis1.sum())) or 1.0
    anchored, lowest_swd = int(np.argmax(val_z)), int(np.argmin([r["val_swd"] for r in rows]))
    mean_of = lambda key: {name: float(np.mean([r[key][name] if key == "test" else r["cluster"][key][name]
                                                for r in rows]))
                           for name in ("r_painting", "judge_z", "distinct")}
    summary = {
        "anchored_seed": anchored, "lowest_val_swd_seed": lowest_swd,
        "anchored_axis1_sign": float(axis1[anchored]), "majority_axis1_sign": majority,
        "n_majority": int((axis1 == majority).sum()),
        "val_test_judge_z_corr": float(np.corrcoef(val_z, test_z)[0, 1]),
        "anchored_test": rows[anchored]["test"], "lowest_val_swd_test": rows[lowest_swd]["test"],
        "mean_test": mean_of("test"),
        "cluster_mean": {key: mean_of(key) for key in rows[0]["cluster"]},
        "groups": {str(k): {"group_gw": g["group_gw"], "pairing": g["pairings"]["gw"].tolist()} for k, g in groups.items()}}
    log.save_json("results.json", {"summary": summary, "seeds": rows})
    print("summary:", {k: v for k, v in summary.items() if k not in ("groups",)})
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
