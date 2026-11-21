from pathlib import Path

import numpy as np

from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of
from abstraction.pipeline import cli
from abstraction.pipeline.features import build_dino, features_of_dataset

CONFIG_PATH = Path(__file__).with_name("config.yaml")
DINO_DIM = 384
SMOKE_TRACKS, SMOKE_IMAGES = 400, 400


def base_parser(description):
    parser = cli.base_parser(description, CONFIG_PATH,
                             smoke_help="synthetic inputs, randomly initialised small models; checks the code path only")
    parser.add_argument("--embeddings", default=None,
                        help="directory of the CAV-MAE embeddings; overrides cav_mae.embeddings")
    return parser


def run_name(stem, smoke):
    return cli.run_name("s4", stem, smoke)


def smoke_image_names():
    return np.array([f"synthetic_{i:04d}.png" for i in range(SMOKE_IMAGES)])


def load_cav_embeddings(directory):
    """(track ids, audio embeddings (n_tracks, n_windows, d), painting names, image embeddings (n, d))."""
    directory = Path(directory)
    ids, audio = load_shards(directory / "audio")
    with np.load(directory / "art.npz") as data:
        names, images = data["names"], data["features"]
    return np.asarray(ids), audio.astype(np.float32), names, images.astype(np.float32)


def load_dino_art(config, device, smoke=False):
    """DINOv2 features of every painting: (file names, features (n, 384)).

    Same cache file as F1 (paths.dino_art_features).
    """
    if smoke:
        rng = np.random.default_rng(1)
        centres = rng.normal(size=(6, DINO_DIM))
        features = centres[rng.integers(6, size=SMOKE_IMAGES)] + 0.7 * rng.normal(size=(SMOKE_IMAGES, DINO_DIM))
        return smoke_image_names(), features.astype(np.float32)
    cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
    if not cache.exists():
        dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
        features = features_of_dataset(build_dino(config, device, False), dataset, device)
        cache.parent.mkdir(parents=True, exist_ok=True)
        np.savez(cache, names=np.asarray(dataset.image_files), features=features.numpy())
    with np.load(cache) as data:
        return data["names"], data["features"]


def splits_of(identifiers, config):
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    return np.array([split_of(str(identifier), val, test) for identifier in identifiers])


def unit(x):
    return x / np.linalg.norm(x, axis=-1, keepdims=True)


def _s1_data():
    import importlib.util
    spec = importlib.util.spec_from_file_location("s1_data", Path(__file__).parents[1] / "s1" / "data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def cav_inputs(directory, smoke, cav):
    """CAV-MAE audio embeddings per track {id: (n_windows, d)}, image embeddings (n, d), painting names.
    In smoke runs: None (synthetic audio embeddings are made per track) and no images."""
    if smoke:
        return None, None, None
    ids, audio, names, images = load_cav_embeddings(directory)
    return {str(t): a for t, a in zip(ids, audio)}, images, names


def aligned_codes(config, split, cav_audio, smoke, embed_dim=96):
    """MERT codes (n, n_windows, 768) and CAV-MAE audio embeddings (n, n_windows, d) of the tracks of a split
    that have both, in the same order; the windows are the same windows of each track."""
    ids, codes = _s1_data().load_codes(config, split, smoke)
    if cav_audio is None:
        rng = np.random.default_rng(len(split))
        return codes, rng.normal(size=(len(ids), codes.shape[1], embed_dim)).astype(np.float32)
    keep = [i for i, t in enumerate(ids) if str(t) in cav_audio]
    return codes[keep], np.stack([cav_audio[str(ids[i])] for i in keep]).astype(np.float32)


def image_centre(config, images, names, smoke, embed_dim=96):
    """Mean CAV-MAE embedding of the training paintings."""
    if images is None:
        return np.zeros(embed_dim, dtype=np.float32)
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    train = np.array([split_of(str(n), val, test) == "train" for n in names])
    return images[train].mean(axis=0).astype(np.float32)
