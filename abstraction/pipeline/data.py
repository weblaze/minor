import numpy as np
import torch
from torch.utils.data import Dataset

from abstraction.data.codes import apply_code_transform, fit_code_transform
from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of


def load_layers(config, smoke=False):
    """All tracks: (ids, array (n_tracks, n_windows, 13, 768) float16, split name per track)."""
    ids, layers = synthetic_layers() if smoke else load_shards(config["paths"]["mert_embeddings"])
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    splits = np.array([split_of(track_id, val, test) for track_id in ids])
    return np.asarray(ids), layers, splits


def codes_of(layers, splits, variant, whiten_dim, layer_index=None):
    """Codes of every track (n_tracks, n_windows, d) and the transform fitted on training tracks
    (abstraction.data.codes)."""
    transform = fit_code_transform(layers[splits == "train"], variant, whiten_dim, layer_index)
    return apply_code_transform(layers, transform), transform


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


def synthetic_layers(n_tracks=600, n_windows=3, dim=768, n_clusters=8, seed=0):
    """Clustered track codes spread over 13 layers with different scales and offsets. For smoke runs only."""
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
