import numpy as np
import torch
from torch.utils.data import Dataset

from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of

CODE_VARIANTS = ("mean", "standardised", "whitened")
CHUNK = 512


def load_layers(config, smoke=False):
    """All tracks: (ids, array (n_tracks, n_windows, 13, 768) float16, split name per track)."""
    ids, layers = _synthetic_layers() if smoke else load_shards(config["paths"]["mert_embeddings"])
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    splits = np.array([split_of(track_id, val, test) for track_id in ids])
    return ids, layers, splits


def fit_code_transform(train_layers, variant, whiten_dim=128):
    """Fit the audio-code transform on training tracks only.

    mean          unweighted mean of the 13 layer vectors
    standardised  each layer z-scored per dimension, then averaged; the result is centred
    whitened      standardised, then projected on the top `whiten_dim` principal axes
                  and scaled to unit variance on each axis
    """
    if variant not in CODE_VARIANTS:
        raise ValueError(f"unknown code variant {variant!r}")
    transform = {"variant": variant}
    if variant == "mean":
        return transform
    flat = train_layers.reshape(-1, *train_layers.shape[2:])
    total, total_sq = 0.0, 0.0
    for start in range(0, len(flat), CHUNK):
        chunk = flat[start:start + CHUNK].astype(np.float64)
        total = total + chunk.sum(axis=0)
        total_sq = total_sq + (chunk ** 2).sum(axis=0)
    mean = total / len(flat)
    std = np.sqrt(np.maximum(total_sq / len(flat) - mean ** 2, 0.0)) + 1e-6
    transform["layer_mean"] = mean.astype(np.float32)
    transform["layer_std"] = std.astype(np.float32)
    if variant == "whitened":
        codes = apply_code_transform(train_layers, {**transform, "variant": "standardised"})
        codes = codes.reshape(-1, codes.shape[-1]).astype(np.float64)
        centre = codes.mean(axis=0)
        eigval, eigvec = np.linalg.eigh(np.cov(codes - centre, rowvar=False))
        order = np.argsort(eigval)[::-1][:whiten_dim]
        transform["centre"] = centre.astype(np.float32)
        transform["projection"] = (eigvec[:, order] / np.sqrt(eigval[order] + 1e-8)).astype(np.float32)
    return transform


def apply_code_transform(layers, transform):
    """(n_tracks, n_windows, 13, 768) -> codes (n_tracks, n_windows, d), float32."""
    out = []
    for start in range(0, len(layers), CHUNK):
        chunk = layers[start:start + CHUNK].astype(np.float32)
        if transform["variant"] != "mean":
            chunk = (chunk - transform["layer_mean"]) / transform["layer_std"]
        codes = chunk.mean(axis=2)
        if transform["variant"] == "whitened":
            codes = (codes - transform["centre"]) @ transform["projection"]
        out.append(codes)
    return np.concatenate(out)


def transform_to_state(transform):
    return {k: (v if isinstance(v, str) else torch.from_numpy(v)) for k, v in transform.items()}


def transform_from_state(state):
    return {k: (v if isinstance(v, str) else v.cpu().numpy()) for k, v in state.items()}


def sample_windows(codes, batch_size, generator):
    """One randomly chosen window from each of `batch_size` distinct tracks."""
    tracks = torch.randperm(codes.shape[0], generator=generator)[:batch_size]
    windows = torch.randint(codes.shape[1], (len(tracks),), generator=generator)
    return codes[tracks, windows]


def art_dataset(config, split, train, smoke=False):
    if smoke:
        return SyntheticImages(160 if split == "train" else 96, config["image_size"], seed=len(split))
    return ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=train, split=split,
                        val_percent=config["split"]["val_percent"],
                        test_percent=config["split"]["test_percent"])


def _synthetic_layers(n_tracks=600, n_windows=3, dim=768, n_clusters=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    tracks = centres[rng.integers(n_clusters, size=n_tracks)] + 0.5 * rng.normal(size=(n_tracks, dim))
    codes = tracks[:, None, :] + 0.1 * rng.normal(size=(n_tracks, n_windows, dim))
    scale = rng.uniform(0.2, 5.0, size=(13, 1))
    offset = rng.normal(size=(13, dim))
    layers = codes[:, :, None, :] * scale + offset
    return [f"synthetic_{i:05d}" for i in range(n_tracks)], layers.astype(np.float16)


class SyntheticImages(Dataset):
    """Random low-frequency colour fields. For smoke runs only."""

    def __init__(self, n, size, seed):
        g = torch.Generator().manual_seed(seed)
        low = torch.rand(n, 3, 4, 4, generator=g) * 2 - 1
        self.images = torch.nn.functional.interpolate(low, size=(size, size), mode="bicubic").clamp(-1, 1)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx]
