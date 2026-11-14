"""Stage 0 diagnostics for Q1 (D1 to D3). No network is trained. Test split only,
with the audio-code transforms fitted on the training split.

D1  Audio codes. For each code variant (mean, standardised, whitened):
    top-1 accuracy of retrieving window 1 of a track from window 0 of the same
    track among all test tracks (chance 1 / n), and the spread of the cosine
    distances between track-mean codes (quantiles, coefficient of variation).
    The variant with the highest retrieval accuracy is reported as `selected`;
    this selection uses no labels.
D2  Image features. Top-1 accuracy of retrieving an artwork from an augmented
    view of it (horizontal flip and 80 % centre crop) in DINOv2 feature space,
    distance spread and participation ratio.
D3  Comparison of the two geometries, for the code variant of the config:
    quantiles of the distances divided by their mean, Wasserstein-1 distance
    between the two normalised distance distributions, fraction of variance on
    the leading principal axes, participation ratios.

D4 is gate.py; D5 is baselines.py.

Usage:
  python methods/q1/diagnostics.py
"""
import numpy as np
import torch.nn.functional as F
from scipy.stats import wasserstein_distance

from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix, retrieval_accuracy
from abstraction.utils.runlog import RunLog, set_seed

from common import base_parser, build_dino, features_of_dataset, run_name, setup
from data import CODE_VARIANTS, apply_code_transform, art_dataset, fit_code_transform, load_layers

QUANTILES = (0.05, 0.25, 0.5, 0.75, 0.95)


def distances(x):
    d = distance_matrix(x)
    return d[np.triu_indices(len(d), k=1)]


def spread(d):
    return {"mean": float(d.mean()), "std": float(d.std()), "coefficient_of_variation": float(d.std() / d.mean()),
            "quantiles": {str(q): float(np.quantile(d, q)) for q in QUANTILES}}


def variance_fractions(x, k=32):
    x = np.asarray(x, dtype=np.float64)
    s = np.linalg.svd(x - x.mean(0), compute_uv=False) ** 2
    return (s[:k] / s.sum()).tolist()


def augment(images):
    size = images.shape[-1]
    margin = size // 10
    cropped = images.flip(3)[:, :, margin:size - margin, margin:size - margin]
    return F.interpolate(cropped, size=(size, size), mode="bilinear", align_corners=False)


def main():
    args = base_parser(__doc__).parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    log = RunLog(config["paths"]["runs_dir"], run_name("diagnostics", args.smoke), config, seed)

    _, layers, splits = load_layers(config, args.smoke)
    train_layers, test_layers = layers[splits == "train"], layers[splits == "test"]
    audio, track_codes = {}, {}
    for variant in CODE_VARIANTS:
        transform = fit_code_transform(train_layers, variant, config["audio_code"]["whiten_dim"])
        codes = apply_code_transform(test_layers, transform)
        track_codes[variant] = codes.mean(axis=1)
        audio[variant] = {
            "code_dim": int(codes.shape[-1]),
            "same_track_retrieval": retrieval_accuracy(codes[:, 0], codes[:, 1]),
            "distance_spread": spread(distances(track_codes[variant])),
            "participation_ratio": participation_ratio(track_codes[variant]),
        }
    selected = max(CODE_VARIANTS, key=lambda v: audio[v]["same_track_retrieval"])

    dino = build_dino(config, device, args.smoke)
    test_set = art_dataset(config, "test", False, args.smoke)
    art = features_of_dataset(dino, test_set, device).numpy()
    art_augmented = features_of_dataset(dino, test_set, device, transform=augment).numpy()

    configured = config["audio_code"]["variant"]
    d_audio, d_image = distances(track_codes[configured]), distances(art)
    result = {
        "D1_audio": {"n_test_tracks": int(len(test_layers)), "chance": 1.0 / len(test_layers),
                     "variants": audio, "selected": selected, "configured": configured},
        "D2_image": {"n_test_art": int(len(art)), "chance": 1.0 / len(art),
                     "augmented_view_retrieval": retrieval_accuracy(art_augmented, art),
                     "distance_spread": spread(d_image),
                     "participation_ratio": participation_ratio(art)},
        "D3_comparison": {
            "code_variant": configured,
            "normalised_quantiles_audio": {str(q): float(np.quantile(d_audio / d_audio.mean(), q)) for q in QUANTILES},
            "normalised_quantiles_image": {str(q): float(np.quantile(d_image / d_image.mean(), q)) for q in QUANTILES},
            "wasserstein_1_normalised_distances": float(wasserstein_distance(d_audio / d_audio.mean(),
                                                                             d_image / d_image.mean())),
            "variance_fractions_audio": variance_fractions(track_codes[configured]),
            "variance_fractions_image": variance_fractions(art),
        },
        "smoke": bool(args.smoke),
    }
    log.save_json("diagnostics.json", result)
    print(f"run directory: {log.dir}")
    for variant in CODE_VARIANTS:
        print(variant, "same-track retrieval", audio[variant]["same_track_retrieval"])
    print("selected:", selected, "| configured:", configured)


if __name__ == "__main__":
    main()
