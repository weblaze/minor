"""Audio codes derived from MERT layer vectors.

A code transform maps the array (n_tracks, n_windows, 13, 768) to codes
(n_tracks, n_windows, d). All statistics are fitted on training tracks.

  mean          unweighted mean of the selected layer vectors
  standardised  each layer z-scored per dimension, then averaged
  whitened      standardised, centred, projected on the leading `whiten_dim`
                principal axes and scaled to unit variance on each axis

`layers` restricts the transform to a subset of the 13 layers (default: all).
"""
import numpy as np
import torch

CODE_VARIANTS = ("mean", "standardised", "whitened")
CHUNK = 512


def fit_code_transform(train_layers, variant, whiten_dim=128, layers=None):
    if variant not in CODE_VARIANTS:
        raise ValueError(f"unknown code variant {variant!r}")
    index = np.arange(train_layers.shape[2]) if layers is None else np.asarray(layers, dtype=np.int64)
    transform = {"variant": variant, "layers": index}
    if variant == "mean":
        return transform
    flat = train_layers.reshape(-1, *train_layers.shape[2:])
    total, total_sq = 0.0, 0.0
    for start in range(0, len(flat), CHUNK):
        chunk = flat[start:start + CHUNK][:, index].astype(np.float64)
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
    out = []
    for start in range(0, len(layers), CHUNK):
        chunk = layers[start:start + CHUNK][:, :, transform["layers"]].astype(np.float32)
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
