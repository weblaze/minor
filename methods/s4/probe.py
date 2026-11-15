"""S4 probe: what a model of natural audio-visual co-occurrence says about the tracks and the paintings.

Nothing is trained. From the CAV-MAE embeddings written by `extract.py` the script reports

  encoder_agreement   agreement of distances between CAV-MAE audio embeddings and MERT codes over the
                      test tracks, and between CAV-MAE image embeddings and DINOv2 features over the
                      test paintings (Mantel correlation, neighbour overlap, trustworthiness)
  cross_modal         statistics of the cosine similarity S between every track and every painting:
                      mean and spread, similarity within each modality, share of the variance of S
                      carried by the track, by the painting and by their interaction, and the
                      correlation of the interaction rows of two windows of the same track against
                      two windows of different tracks
  painting            S4 used as a bridge without training: each test track receives the training
                      painting with the highest S. Scored like the painting block of F1 (MERT codes
                      against DINOv2 features of the assigned paintings)
  painting_centred    the same after removing from S the mean of each painting over the training
                      tracks and dividing by its standard deviation
  d_s4                for each `--assignments` file (features.npz of an evaluation that assigns a
                      painting to each track): mean S over the assigned pairs against the same
                      paintings re-assigned at random to the same tracks

Genre labels are used only by `--genres`, for a linear probe on the frozen embeddings.

Usage:
  python methods/s4/probe.py [--embeddings DIR] [--assignments runs/f1_eval_*/features.npz ...]
"""
import glob

import numpy as np

from abstraction.pipeline.cli import setup
from abstraction.pipeline.data import codes_of, load_layers
from abstraction.pipeline.scores import full_scores, genre_leakage, structure_scores
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, load_cav_embeddings, load_dino_art, run_name, splits_of, unit

CODES = ("mean", "standardised")


def align(wanted, available):
    """Indices into `available` for the entries of `wanted` present in it, and the mask over `wanted`."""
    position = {str(name): i for i, name in enumerate(available)}
    present = np.array([str(name) in position for name in wanted])
    return np.array([position[str(name)] for name in np.asarray(wanted)[present]], dtype=int), present


def variance_shares(s):
    """Shares of the variance of a matrix carried by row means, column means and the remainder."""
    rows, columns = s.mean(axis=1, keepdims=True), s.mean(axis=0, keepdims=True)
    total = s.var()
    return {"track": float(rows.var() / total), "painting": float(columns.var() / total),
            "interaction": float((s - rows - columns + s.mean()).var() / total)}


def double_centre(s):
    return s - s.mean(axis=1, keepdims=True) - s.mean(axis=0, keepdims=True) + s.mean()


def row_correlations(a, b):
    a = a - a.mean(axis=1, keepdims=True)
    b = b - b.mean(axis=1, keepdims=True)
    return (a * b).sum(axis=1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def window_consistency(audio_windows, images, rng):
    """Interaction rows of S computed from window 0 and from window 1: same track against other tracks."""
    first = double_centre(unit(audio_windows[:, 0]) @ images.T)
    second = double_centre(unit(audio_windows[:, 1]) @ images.T)
    other = rng.permutation(len(first))
    other[other == np.arange(len(first))] = (other[other == np.arange(len(first))] + 1) % len(first)
    return {"same_track": float(row_correlations(first, second).mean()),
            "other_track": float(row_correlations(first, second[other]).mean())}


def mean_off_diagonal(x):
    s = x @ x.T
    return float((s.sum() - np.trace(s)) / (len(s) * (len(s) - 1)))


def concentration(nearest):
    _, counts = np.unique(nearest, return_counts=True)
    return {"n_distinct_paintings": int(len(counts)), "largest_share": float(counts.max() / len(nearest))}


def assignment_score(s, tracks, paintings, n_null, rng):
    """Mean of S over assigned pairs, and over random re-assignments of the same paintings."""
    observed = float(s[tracks, paintings].mean())
    null = np.array([s[tracks, rng.permutation(paintings)].mean() for _ in range(n_null)])
    return {"n_pairs": int(len(tracks)), "mean_similarity": observed, "null_mean": float(null.mean()),
            "null_std": float(null.std()), "z": float((observed - null.mean()) / null.std()),
            "p_greater": float((1 + (null >= observed).sum()) / (1 + n_null))}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--assignments", nargs="*", default=[], help="features.npz files with track_ids and painting_names")
    parser.add_argument("--genres", default=None, help="CSV with track_id and genre; adds a linear genre probe")
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    rng = np.random.default_rng(seed)
    hp = config["probe"]
    n_perm = args.permutations or (199 if args.smoke else hp["n_permutations"])
    n_null = 99 if args.smoke else hp["n_null"]
    embeddings = args.embeddings or config["cav_mae"]["embeddings"]
    config["run"] = {"embeddings": embeddings, "assignments": args.assignments}
    log = RunLog(config["paths"]["runs_dir"], run_name("eval_probe", args.smoke), config, seed)

    cav_ids, cav_audio, cav_names, cav_images = load_cav_embeddings(embeddings)
    ids, layers, splits = load_layers(config, args.smoke)
    keep, present = align(ids, cav_ids)
    ids, layers, splits, cav_audio = ids[present], layers[present], splits[present], cav_audio[keep]
    names, dino = load_dino_art(config, device, args.smoke)
    keep, present = align(names, cav_names)
    names, dino, cav_images = names[present], dino[present], cav_images[keep]
    art_splits = splits_of(names, config)
    test, train = splits == "test", splits == "train"
    art_test, art_train = art_splits == "test", art_splits == "train"

    audio, images = unit(cav_audio.mean(axis=1)), unit(cav_images)
    results = {"n_tracks": int(len(ids)), "n_paintings": int(len(names)),
               "n_test_tracks": int(test.sum()), "n_test_paintings": int(art_test.sum())}

    code = config["audio_code"]
    mert = {variant: codes_of(layers, splits, variant, code["whiten_dim"], code["layers"])[0].mean(axis=1)
            for variant in CODES}
    results["encoder_agreement"] = {
        "audio": {variant: structure_scores(mert[variant][test], audio[test], hp["knn_k"], n_perm, seed)
                  for variant in CODES},
        "image": structure_scores(dino[art_test], images[art_test], hp["knn_k"], n_perm, seed)}
    if args.genres:
        probe = genre_leakage(ids[test].tolist(), audio[test], mert["standardised"][test], args.genres)
        probe["accuracy_from_cav_mae_audio"] = probe.pop("accuracy_from_images")
        probe["accuracy_from_mert"] = probe.pop("accuracy_from_audio")
        results["genre_probe"] = probe

    s = audio @ images.T
    results["cross_modal"] = {
        "mean": float(s.mean()), "std": float(s.std()), "min": float(s.min()), "max": float(s.max()),
        "mean_within_audio": mean_off_diagonal(audio[test]),
        "mean_within_image": mean_off_diagonal(images[art_test]),
        "centroid_cosine": float(unit(audio.mean(axis=0)) @ unit(images.mean(axis=0))),
        "variance_shares": variance_shares(s),
        "window_consistency": window_consistency(cav_audio[test], images[art_train], rng)}

    gallery = np.flatnonzero(art_train)
    s_gallery = s[:, gallery]
    centred = (s_gallery - s_gallery[train].mean(axis=0)) / s_gallery[train].std(axis=0)
    primary_code = mert[code["variant"]] if code["variant"] in mert else mert["standardised"]
    saved = {}
    for key, matrix in (("painting", s_gallery), ("painting_centred", centred)):
        nearest = gallery[matrix[test].argmax(axis=1)]
        results[key] = {**full_scores(primary_code[test], dino[nearest], dino[art_test], hp["knn_k"], n_perm, seed),
                        **concentration(nearest)}
        saved[key] = nearest
    results["code_variant"] = code["variant"] if code["variant"] in mert else "standardised"

    paths = sorted(path for pattern in args.assignments for path in glob.glob(pattern))
    track_position = {str(t): i for i, t in enumerate(ids)}
    name_position = {str(n): i for i, n in enumerate(names)}
    gallery_position = {int(g): i for i, g in enumerate(gallery)}
    results["d_s4"] = {}
    for path in paths:
        with np.load(path) as data:
            if "painting_names" not in data:
                continue
            pairs = [(track_position[str(t)], name_position[str(n)])
                     for t, n in zip(data["track_ids"], data["painting_names"])
                     if str(t) in track_position and str(n) in name_position]
        tracks, paintings = (np.array(x) for x in zip(*pairs))
        entry = {"similarity": assignment_score(s, tracks, paintings, n_null, rng)}
        if all(int(p) in gallery_position for p in paintings):
            columns = np.array([gallery_position[int(p)] for p in paintings])
            entry["similarity_centred"] = assignment_score(centred, tracks, columns, n_null, rng)
        results["d_s4"][path.replace("\\", "/")] = entry

    log.save_json("results.json", results)
    np.savez(log.dir / "features.npz", track_ids=ids[test], audio_codes=primary_code[test],
             painting_features=dino[saved["painting"]], painting_names=names[saved["painting"]],
             painting_names_centred=names[saved["painting_centred"]], cav_audio=audio[test])
    for key in ("encoder_agreement", "cross_modal"):
        print(key, results[key])
    for key in ("painting", "painting_centred"):
        block = results[key]
        print(key, block["structure"], "FD", block["image_distribution"]["frechet_distance"],
              "distinct", block["n_distinct_paintings"], "largest share", block["largest_share"])
    for path, entry in results["d_s4"].items():
        print(path, entry)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
