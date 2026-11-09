import numpy as np
import torch
from torch.utils.data import Dataset

from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of


def load_codes(config, split, smoke=False):
    """Audio codes of one split: (track ids, array (n_tracks, n_windows, 768)).

    The code of a window is the unweighted mean of the 13 MERT layer vectors.
    """
    if smoke:
        ids, codes = _synthetic_codes()
    else:
        ids, layers = load_shards(config["paths"]["mert_embeddings"])
        codes = layers.astype(np.float32).mean(axis=2)
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    keep = [i for i, track_id in enumerate(ids) if split_of(track_id, val, test) == split]
    return [ids[i] for i in keep], codes[keep]


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


def _synthetic_codes(n_tracks=600, n_windows=3, dim=768, n_clusters=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    tracks = centres[rng.integers(n_clusters, size=n_tracks)] + 0.5 * rng.normal(size=(n_tracks, dim))
    codes = tracks[:, None, :] + 0.1 * rng.normal(size=(n_tracks, n_windows, dim))
    return [f"synthetic_{i:05d}" for i in range(n_tracks)], codes.astype(np.float32)


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
