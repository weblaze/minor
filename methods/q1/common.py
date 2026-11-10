import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.manifold import trustworthiness
from torch.utils.data import DataLoader

from abstraction.encoders.dino import DinoFeatures
from abstraction.metrics.distribution import frechet_distance, kernel_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix, knn_overlap, mantel
from abstraction.models.mapper import Mapper
from abstraction.models.vector_vae import VectorVAE
from abstraction.utils.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")


def base_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="appended to the run directory name")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic data, randomly initialised DINOv2, few steps; checks the code path only")
    return parser


def setup(args):
    config = load_config(args.config)
    if args.seed is not None:
        config["system"]["seed"] = args.seed
    if args.tag:
        config["system"]["tag"] = args.tag
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    return config, device


def run_name(stem, smoke):
    return f"q1_{stem}" + ("_smoke" if smoke else "")


def build_dino(config, device, smoke):
    if smoke:
        return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"], pretrained=False, depth=2).to(device)
    return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)


def build_vae(config):
    return VectorVAE(config["vae"]["latent_dim"], config["vae"]["base_channels"])


def load_vae(path, config, device):
    state = torch.load(path, map_location=device, weights_only=True)
    vae = build_vae(config).to(device)
    vae.load_state_dict(state["model"])
    vae.eval()
    for p in vae.parameters():
        p.requires_grad = False
    return vae


def gate_passed(vae_path):
    """True, False, or None if no gate result exists next to the VAE checkpoint."""
    path = Path(vae_path).with_name("gate.json")
    if not path.exists():
        return None
    with open(path) as f:
        return bool(json.load(f)["passed"])


def build_mapper(config, code_dim):
    return Mapper(code_dim, config["vae"]["latent_dim"], config["mapper"]["hidden"], config["mapper"]["depth"])


@torch.no_grad()
def encode_images(vae, dataset, device, batch_size=64):
    """Posterior means and log-variances of every image of a dataset, on the CPU."""
    mus, logvars = [], []
    for images in DataLoader(dataset, batch_size=batch_size):
        mu, logvar = vae.encode(images.to(device))
        mus.append(mu.float().cpu())
        logvars.append(logvar.float().cpu())
    return torch.cat(mus), torch.cat(logvars)


@torch.no_grad()
def features_of_dataset(dino, dataset, device, batch_size=64, transform=None):
    """DINOv2 features of every image of a dataset, optionally after `transform`."""
    out = []
    for images in DataLoader(dataset, batch_size=batch_size):
        images = images.to(device)
        if transform is not None:
            images = transform(images)
        out.append(dino(images).float().cpu())
    return torch.cat(out)


@torch.no_grad()
def features_of_latents(dino, vae, latents, device, batch_size=64):
    """Decode latents and return (DINOv2 features, first batch of images)."""
    features, first = [], None
    for start in range(0, len(latents), batch_size):
        images = vae.decode(latents[start:start + batch_size].to(device))
        features.append(dino(images).float().cpu())
        if first is None:
            first = images.cpu()
    return torch.cat(features), first


def code_statistics(train_codes):
    """Per-dimension mean and standard deviation of training codes (n_tracks, n_windows, d)."""
    flat = train_codes.reshape(-1, train_codes.shape[-1])
    return flat.mean(0), flat.std(0) + 1e-6


def generate(mapper, vae, codes, code_mean, code_std):
    """Audio codes (b, d) -> (images in [-1, 1], latents (b, latent_dim))."""
    z = mapper((codes - code_mean) / code_std)
    return vae.decode(z), z


def principal_axes(x, k):
    """Mean, top-k principal axes (d, k) and standard deviations along them."""
    x = np.asarray(x, dtype=np.float64)
    mean = x.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(x - mean, rowvar=False))
    order = np.argsort(eigval)[::-1][:k]
    return mean, eigvec[:, order], np.sqrt(np.maximum(eigval[order], 1e-12))


def distribution_scores(features, reference, seed=0):
    kid_mean, kid_std = kernel_distance(features, reference, seed=seed)
    return {"n": len(features), "n_reference": len(reference),
            "frechet_distance": frechet_distance(features, reference),
            "kernel_distance": kid_mean, "kernel_distance_std": kid_std}


def mean_pairwise_distance(features):
    d = distance_matrix(features)
    return float(d[np.triu_indices(len(d), k=1)].mean())


def structure_scores(codes, features, k, n_permutations, seed):
    """Agreement between cosine distances of audio codes and of image features over the same tracks."""
    d_audio, d_image = distance_matrix(codes), distance_matrix(features)
    r_pearson, p_pearson = mantel(d_audio, d_image, "pearson", n_permutations, seed)
    r_spearman, p_spearman = mantel(d_audio, d_image, "spearman", n_permutations, seed)
    overlap, chance = knn_overlap(d_audio, d_image, k)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    return {
        "n_tracks": len(codes), "n_permutations": n_permutations,
        "mantel_pearson": r_pearson, "mantel_pearson_p": p_pearson,
        "mantel_spearman": r_spearman, "mantel_spearman_p": p_spearman,
        "knn_k": k, "knn_overlap": overlap, "knn_overlap_chance": chance,
        "trustworthiness": float(trustworthiness(d_audio, unit, n_neighbors=k, metric="precomputed")),
    }


def score_outputs(codes, features, test_art, config, n_permutations, seed):
    """Full evaluation of one set of outputs: structure, image distribution, collapse indicators."""
    return {
        "structure": structure_scores(codes, features, config["eval"]["knn_k"], n_permutations, seed),
        "image_distribution": {
            **distribution_scores(features, test_art, seed),
            "mean_pairwise_distance": mean_pairwise_distance(features),
            "mean_pairwise_distance_test_art": mean_pairwise_distance(test_art),
            "participation_ratio": participation_ratio(features),
            "participation_ratio_test_art": participation_ratio(test_art),
            "participation_ratio_audio": participation_ratio(codes),
        },
    }
