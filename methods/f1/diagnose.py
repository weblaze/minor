"""F1 diagnosis: where structure is lost between the audio code and a set of real paintings.

For every audio code listed in diagnose.codes a mapping is trained (variant f1) and scored at
the following stages. r is the Pearson correlation between cosine distances of audio codes and
of image features; all scores are on window-averaged codes.

  train_direct      outputs of M for a random subset of training tracks (same size as the test set)
  test_direct       outputs of M for the test tracks; the difference to train_direct is the
                    generalisation gap of M
  test_painting     each test output replaced by the nearest training painting; the difference to
                    test_direct is the loss from restricting outputs to real paintings
  gallery_<n>       as test_painting with a random subset of n training paintings; shows whether the
                    restriction loss comes from the number of available paintings
  oracle_painting   a mapping trained on the test tracks themselves, then nearest painting; an upper
                    reference for what map-then-retrieve can reach for these tracks and paintings

The code itself is described by its participation ratio and the coefficient of variation of its
pairwise distances on the test tracks.

Usage:
  python methods/f1/diagnose.py --seed 0
"""
import numpy as np

from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix
from abstraction.utils.runlog import RunLog, set_seed

from common import (base_parser, codes_of, load_art, load_layers, map_codes, nearest_paintings, quick_scores,
                    run_name, setup, train_mapper)


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    steps = args.steps or (60 if args.smoke else config["mapper"]["steps"])
    k = config["eval"]["knn_k"]
    set_seed(seed)
    config["run"] = {"steps": steps}
    log = RunLog(config["paths"]["runs_dir"], run_name("diagnose", args.smoke), config, seed)

    _, layers, splits = load_layers(config, args.smoke)
    _, art, art_splits = load_art(config, device, args.smoke)
    gallery, art_val, test_art = (art[art_splits == s] for s in ("train", "val", "test"))
    rng = np.random.default_rng(seed)
    n_test = int((splits == "test").sum())
    train_subset = rng.choice(int((splits == "train").sum()), size=n_test, replace=False)
    sizes = [n for n in config["diagnose"]["gallery_sizes"] if n < len(gallery)]
    subsets = {n: rng.choice(len(gallery), size=n, replace=False) for n in sizes}

    results = {}
    for name, variant, layer_index in config["diagnose"]["codes"]:
        codes, _ = codes_of(layers, splits, variant, config["audio_code"]["whiten_dim"], layer_index)
        train, val, test = (codes[splits == s] for s in ("train", "val", "test"))
        test_mean, train_mean = test.mean(axis=1), train[train_subset].mean(axis=1)
        d_test = distance_matrix(test_mean)
        iu = np.triu_indices(len(d_test), k=1)
        row = {"code": {"dim": int(codes.shape[-1]), "participation_ratio": participation_ratio(test_mean),
                        "distance_cv": float(d_test[iu].std() / d_test[iu].mean())}}

        mapper, mean, std = train_mapper(config, train, val, gallery, art_val, device, "f1", steps, seed)
        direct = map_codes(mapper, test_mean, mean, std, device)
        row["train_direct"] = quick_scores(train_mean, map_codes(mapper, train_mean, mean, std, device), test_art, k)
        row["test_direct"] = quick_scores(test_mean, direct, test_art, k)
        nearest = nearest_paintings(direct, gallery)
        row["test_painting"] = quick_scores(test_mean, gallery[nearest], test_art, k)
        row["test_painting"]["n_distinct_paintings"] = int(len(np.unique(nearest)))
        for n, subset in subsets.items():
            sub = gallery[subset]
            row[f"gallery_{n}"] = quick_scores(test_mean, sub[nearest_paintings(direct, sub)], test_art, k)

        oracle, mean, std = train_mapper(config, test, test, gallery, art_val, device, "f1", steps, seed)
        nearest = nearest_paintings(map_codes(oracle, test_mean, mean, std, device), gallery)
        row["oracle_painting"] = quick_scores(test_mean, gallery[nearest], test_art, k)
        row["oracle_painting"]["n_distinct_paintings"] = int(len(np.unique(nearest)))

        results[name] = row
        log.save_json("diagnosis.json", results)
        stages = ["train_direct", "test_direct", "test_painting"] + [f"gallery_{n}" for n in sizes] + ["oracle_painting"]
        print(f"{name}: PR {row['code']['participation_ratio']:.1f}, CV {row['code']['distance_cv']:.3f} | "
              + " | ".join(f"{s} r {row[s]['r']:.3f} FD {row[s]['frechet_distance']:.0f}" for s in stages), flush=True)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
