import numpy as np

from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of


def load_codes(config, split, smoke=False):
    """Audio codes of one split: (track ids, array (n_tracks, n_windows, d)).

    mean          unweighted mean of the 13 MERT layer vectors (d = 768)
    standardised  each layer z-scored per dimension, then averaged (d = 768); fitted on training tracks
    whitened      standardised, centred, projected on the leading `whiten_dim` principal axes and
                  scaled to unit variance

    Kept apart from abstraction.data.codes: these float64 statistics differ from the shared
    transform in the last bits, so replacing it would change S1 runs on the standardised and
    whitened codes.
    """
    if smoke:
        ids, codes = _synthetic_codes()
        layers = codes[:, :, None, :]
    else:
        ids, layers = load_shards(config["paths"]["mert_embeddings"])
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    splits = np.array([split_of(track_id, val, test) for track_id in ids])
    keep = np.flatnonzero(splits == split)
    variant = config["audio_code"]["variant"]
    if variant == "mean":
        codes = layers[keep].astype(np.float32).mean(axis=2)
    elif variant in ("standardised", "whitened"):
        dim = config["audio_code"]["whiten_dim"] if variant == "whitened" else None
        codes = _standardised(layers[splits == "train"], layers[keep], dim)
    else:
        raise ValueError(f"unknown code variant {variant!r}")
    return [ids[i] for i in keep], codes


def _standardised(train_layers, layers, whiten_dim=None):
    flat = train_layers.reshape(-1, *train_layers.shape[2:])
    mean = flat.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = flat.std(axis=0, dtype=np.float64).astype(np.float32) + 1e-6

    def standardised(x):
        return ((x.astype(np.float32) - mean) / std).mean(axis=2)

    if whiten_dim is None:
        return standardised(layers)
    train = standardised(train_layers).reshape(-1, mean.shape[-1]).astype(np.float64)
    centre = train.mean(axis=0)
    eigval, eigvec = np.linalg.eigh(np.cov(train - centre, rowvar=False))
    order = np.argsort(eigval)[::-1][:whiten_dim]
    projection = eigvec[:, order] / np.sqrt(eigval[order] + 1e-8)
    return ((standardised(layers) - centre) @ projection).astype(np.float32)


def _synthetic_codes(n_tracks=600, n_windows=3, dim=768, n_clusters=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    tracks = centres[rng.integers(n_clusters, size=n_tracks)] + 0.5 * rng.normal(size=(n_tracks, dim))
    codes = tracks[:, None, :] + 0.1 * rng.normal(size=(n_tracks, n_windows, dim))
    return [f"synthetic_{i:05d}" for i in range(n_tracks)], codes.astype(np.float32)

