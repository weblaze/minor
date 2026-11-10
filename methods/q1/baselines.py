"""Trivial baselines for Q1 (diagnostic D5). No network is trained.

Both baselines use one linear map from audio codes to image latents, built from
second-order statistics of the two training sets:

  the i-th principal axis of the training audio codes (track means) is sent to the
  i-th principal axis of the posterior means of training art, for i = 1..n_axes,
  with each coordinate rescaled from the audio standard deviation to the latent
  standard deviation on that axis. Axes are matched by variance rank. The sign of
  each axis is the one returned by the eigendecomposition and is arbitrary.

  pca_alignment  the mapped latent is decoded by the VAE
  retrieval      the output is the real training artwork whose posterior mean is
                 nearest to the mapped latent (Euclidean distance in the n_axes
                 standardised coordinates); no image is generated

Both are scored on the test tracks with the same function as evaluate.py. A
trained mapping has to exceed these scores for its structure result to be
attributed to training.

Usage:
  python methods/q1/baselines.py --vae runs/<vae run>/vae.pt
"""
import numpy as np
import torch
from scipy.spatial.distance import cdist
from torchvision.utils import save_image

from abstraction.utils.runlog import RunLog, set_seed

from common import (base_parser, build_dino, encode_images, features_of_dataset, features_of_latents, load_vae,
                    principal_axes, run_name, score_outputs, setup)
from data import apply_code_transform, art_dataset, fit_code_transform, load_layers


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    parser.add_argument("--permutations", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    n_perm = args.permutations or (199 if args.smoke else config["eval"]["n_permutations"])
    config["run"] = {"vae": str(args.vae)}
    log = RunLog(config["paths"]["runs_dir"], run_name("baselines", args.smoke), config, seed)

    vae = load_vae(args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    _, layers, splits = load_layers(config, args.smoke)
    transform = fit_code_transform(layers[splits == "train"], config["audio_code"]["variant"],
                                   config["audio_code"]["whiten_dim"])
    train_codes = apply_code_transform(layers[splits == "train"], transform).mean(axis=1)
    test_codes = apply_code_transform(layers[splits == "test"], transform).mean(axis=1)
    del layers

    train_set = art_dataset(config, "train", False, args.smoke)
    mu, _ = encode_images(vae, train_set, device)
    mu = mu.numpy()
    train_art = features_of_dataset(dino, train_set, device).numpy()
    test_art = features_of_dataset(dino, art_dataset(config, "test", False, args.smoke), device).numpy()

    k = min(config["baseline"]["n_axes"], train_codes.shape[1], mu.shape[1], len(train_codes) - 1, len(mu) - 1)
    audio_mean, audio_axes, audio_std = principal_axes(train_codes, k)
    latent_mean, latent_axes, latent_std = principal_axes(mu, k)
    coordinates = (test_codes - audio_mean) @ audio_axes / audio_std
    latents = latent_mean + (coordinates * latent_std) @ latent_axes.T
    art_coordinates = (mu - latent_mean) @ latent_axes / latent_std
    nearest = cdist(coordinates, art_coordinates).argmin(axis=1)

    features, images = features_of_latents(dino, vae, torch.from_numpy(latents).float(), device)
    save_image(images * 0.5 + 0.5, log.dir / "pca_alignment.png", nrow=8)
    save_image(torch.stack([train_set[int(i)] for i in nearest[:64]]) * 0.5 + 0.5, log.dir / "retrieval.png", nrow=8)

    results = {
        "n_axes": int(k), "code_variant": transform["variant"],
        "pca_alignment": score_outputs(test_codes, features.numpy(), test_art, config, n_perm, seed),
        "retrieval": {**score_outputs(test_codes, train_art[nearest], test_art, config, n_perm, seed),
                      "n_distinct_artworks": int(len(np.unique(nearest)))},
    }
    log.save_json("results.json", results)
    print(f"run directory: {log.dir}")
    for name in ("pca_alignment", "retrieval"):
        print(name, results[name]["structure"])


if __name__ == "__main__":
    main()
