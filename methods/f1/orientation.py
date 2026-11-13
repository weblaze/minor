"""F1: orientation of the mapping across seeds.

The structure term depends on distances only, so a mapping and its mirror image along any axis
have the same structure loss. This script trains one mapping per seed and records

  sign_k      sign of the correlation between the test tracks' scores on principal axis k of the
              audio code and the scores of their paintings on principal axis k of the art features
  corr_k      that correlation
  r_painting  Pearson correlation of audio and painting distance matrices
  val_swd, val_energy   distribution terms between the outputs for validation tracks and the
              features of validation art (not used in training)
  val_struct  1 - Pearson of distance matrices on validation tracks (direct output)

and, between seeds, the mean cosine similarity of the two paintings assigned to the same track.
If one orientation has a consistently lower validation distribution term, the data determine
the orientation and the lowest-loss seed can be selected without labels.

Skewness of the leading principal axes of both spaces is printed: a distribution that is
symmetric along an axis cannot determine the orientation of that axis.

Usage:
  python methods/f1/orientation.py --code mean --configs swd swd_w5 soft_0.02_swd --seeds 10
"""
import json

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import skew

from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.metrics.structure import distance_matrix
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, codes_of, correlation, load_art, load_layers, run_name, setup
from realism import DEFAULT, GRID, energy_distance_np, fit

EXTRA = {"swd_w5": {"dist": [["swd", 5.0]]}, "swd_w20": {"dist": [["swd", 20.0]]}}
N_AXES = 3


def axes_of(x, k):
    mean = x.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(eigval)[::-1][:k]
    return mean, eigvec[:, order]


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--configs", nargs="+", default=["swd"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    steps = args.steps or (40 if args.smoke else config["mapper"]["steps"])
    set_seed(0)
    config["run"] = {"steps": steps, "configs": args.configs, "seeds": args.seeds}
    log = RunLog(config["paths"]["runs_dir"], run_name("orientation", args.smoke), config, 0)

    _, layers, splits = load_layers(config, args.smoke)
    _, art, art_splits = load_art(config, device, args.smoke)
    gallery_np, val_art = art[art_splits == "train"], art[art_splits == "val"]
    gallery = torch.from_numpy(gallery_np).to(device)
    code = config["audio_code"]
    codes, _ = codes_of(layers, splits, code["variant"], code["whiten_dim"], code["layers"])
    to = lambda x: torch.from_numpy(x).to(device)
    train = to(codes[splits == "train"])
    val, test = to(codes[splits == "val"].mean(axis=1)), to(codes[splits == "test"].mean(axis=1))
    test_np = test.cpu().numpy()
    d_audio = distance_matrix(test_np)

    code_mean, code_axes = axes_of(codes[splits == "train"].mean(axis=1), N_AXES)
    art_mean, art_axes = axes_of(gallery_np, N_AXES)
    audio_scores = (test_np - code_mean) @ code_axes
    print("skewness of the leading axes | audio code:",
          np.round(skew((codes[splits == "train"].mean(axis=1) - code_mean) @ code_axes), 3),
          "| art features:", np.round(skew((gallery_np - art_mean) @ art_axes), 3))

    unit = gallery_np / np.linalg.norm(gallery_np, axis=1, keepdims=True)
    scale = float(gallery_np.std())
    results = {}
    for name in args.configs:
        spec = {**DEFAULT, **{**GRID, **EXTRA}[name]}
        rows, picks = [], []
        for seed in range(args.seeds):
            forward = fit(spec, config, train, [], gallery, device, steps, seed)
            with torch.no_grad():
                _, query = forward(test)
                nearest = (F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T).argmax(1).cpu().numpy()
                out_val, _ = forward(val)
                torch.manual_seed(0)
                val_swd = float(sliced_wasserstein(out_val / scale, to(val_art) / scale, 2048))
                val_struct = float(distance_correlation_loss(pairwise_distances(val), pairwise_distances(out_val)))
            painting_scores = (gallery_np[nearest] - art_mean) @ art_axes
            corr = [float(np.corrcoef(audio_scores[:, k], painting_scores[:, k])[0, 1]) for k in range(N_AXES)]
            rows.append({"seed": seed, "corr": corr, "r_painting": correlation(d_audio, distance_matrix(gallery_np[nearest])),
                         "val_swd": val_swd, "val_energy": energy_distance_np(out_val.cpu().numpy() / scale, val_art / scale),
                         "val_struct": val_struct, "distinct": int(len(np.unique(nearest)))})
            picks.append(nearest)
            results[name] = {"seeds": rows}
            log.save_json("results.json", results)
        sims = [float((unit[picks[a]] * unit[picks[b]]).sum(1).mean())
                for a in range(args.seeds) for b in range(a + 1, args.seeds)]
        same_sign = [float((unit[picks[a]] * unit[picks[b]]).sum(1).mean())
                     for a in range(args.seeds) for b in range(a + 1, args.seeds)
                     if np.sign(rows[a]["corr"][0]) == np.sign(rows[b]["corr"][0])]
        rng = np.random.default_rng(0)
        results[name]["between_seeds"] = {
            "painting_similarity": float(np.mean(sims)),
            "painting_similarity_same_orientation": float(np.mean(same_sign)) if same_sign else None,
            "painting_similarity_shuffled": float((unit[picks[0]] * unit[rng.permutation(picks[1])]).sum(1).mean())}
        log.save_json("results.json", results)
        print(f"== {name} ({code['variant']} code)")
        print("seed | corr axis 1, 2, 3 | r painting | val swd | val energy | val struct | distinct")
        for row in rows:
            print(f"{row['seed']} | " + ", ".join(f"{c:+.2f}" for c in row["corr"])
                  + f" | {row['r_painting']:.3f} | {row['val_swd']:.4f} | {row['val_energy']:.4f} | {row['val_struct']:.4f} | {row['distinct']}")
        print(json.dumps(results[name]["between_seeds"]), flush=True)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
