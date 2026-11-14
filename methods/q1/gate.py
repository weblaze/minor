"""Sampleability gate for the Q1 generator (diagnostic D4).

A mapping can only produce images that the decoder produces for latents away
from the encodings of training images. This script measures that directly.
Four sets of latents are decoded and compared with test-split art by Frechet
distance and kernel distance on DINOv2 features:

  reconstructions  posterior means of training images
  prior            samples from N(0, I)
  interpolations   midpoints between the posterior means of random pairs of training images
  gmm              samples from a Gaussian mixture fitted to the posterior means

Reference: training art against test art.

Also reported: per-dimension statistics of the aggregate posterior and the
number of active latent dimensions (variance of the posterior mean > 0.01).

Decision rule, fixed before any run:
  passed = FD(prior) <= gate.max_fd_ratio * FD(reconstructions)

The result is written to gate.json next to the VAE checkpoint.

Usage:
  python methods/q1/gate.py --vae runs/<vae run>/vae.pt
"""
import json
from pathlib import Path

import torch
from sklearn.mixture import GaussianMixture
from torchvision.utils import save_image

from abstraction.utils.runlog import RunLog, set_seed

from common import (base_parser, build_dino, distribution_scores, encode_images, features_of_dataset,
                    features_of_latents, load_vae, run_name, setup)
from data import art_dataset


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--vae", required=True)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    hp = config["gate"]
    n = 128 if args.smoke else hp["n_samples"]
    config["run"] = {"vae": str(args.vae)}
    log = RunLog(config["paths"]["runs_dir"], run_name("gate", args.smoke), config, seed)

    vae = load_vae(args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    train_set = art_dataset(config, "train", False, args.smoke)
    mu, _ = encode_images(vae, train_set, device)
    test_art = features_of_dataset(dino, art_dataset(config, "test", False, args.smoke), device).numpy()
    train_art = features_of_dataset(dino, train_set, device).numpy()

    g = torch.Generator().manual_seed(seed)
    first = torch.randint(len(mu), (n,), generator=g)
    second = torch.randint(len(mu), (n,), generator=g)
    gmm = GaussianMixture(min(hp["gmm_components"], len(mu) // 10), covariance_type="diag", random_state=seed)
    gmm.fit(mu.numpy())
    latents = {
        "reconstructions": mu,
        "prior": torch.randn(n, mu.shape[1], generator=g),
        "interpolations": 0.5 * (mu[first] + mu[second]),
        "gmm": torch.from_numpy(gmm.sample(n)[0]).float(),
    }

    result = {"reference_train_art": distribution_scores(train_art, test_art, seed)}
    for name, z in latents.items():
        features, images = features_of_latents(dino, vae, z, device)
        result[name] = distribution_scores(features.numpy(), test_art, seed)
        save_image(images[:32] * 0.5 + 0.5, log.dir / f"{name}.png", nrow=8)

    variance = mu.var(dim=0)
    result["aggregate_posterior"] = {
        "mean_abs_mean": float(mu.mean(dim=0).abs().mean()),
        "mean_variance_of_mu": float(variance.mean()),
        "active_dimensions": int((variance > 0.01).sum()),
        "latent_dim": int(mu.shape[1]),
    }
    ratio = result["prior"]["frechet_distance"] / result["reconstructions"]["frechet_distance"]
    result["fd_ratio_prior_to_reconstructions"] = ratio
    result["max_fd_ratio"] = hp["max_fd_ratio"]
    result["passed"] = bool(ratio <= hp["max_fd_ratio"])
    result["smoke"] = bool(args.smoke)

    log.save_json("gate.json", result)
    with open(Path(args.vae).with_name("gate.json"), "w") as f:
        json.dump(result, f, indent=2)
    print(f"run directory: {log.dir}")
    print(f"FD prior / FD reconstructions = {ratio:.3f}; passed = {result['passed']}")


if __name__ == "__main__":
    main()
