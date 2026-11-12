import argparse
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps
from sklearn.manifold import trustworthiness
from torch.utils.data import DataLoader

from abstraction.data.codes import apply_code_transform, fit_code_transform
from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import load_shards
from abstraction.data.splits import split_of
from abstraction.encoders.dino import DinoFeatures
from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.metrics.distribution import frechet_distance, kernel_distance
from abstraction.metrics.geometry import participation_ratio
from abstraction.metrics.structure import distance_matrix, knn_overlap, mantel
from abstraction.models.mapper import Mapper
from abstraction.utils.config import load_config

CONFIG_PATH = Path(__file__).with_name("config.yaml")
FEATURE_DIM = 384
VARIANTS = {"f1": (True, False), "f0": (False, False), "shuffled": (True, True)}


def base_parser(description):
    parser = argparse.ArgumentParser(description=description,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(CONFIG_PATH))
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--tag", default=None, help="appended to the run directory name")
    parser.add_argument("--code", choices=("mean", "standardised", "whitened"), default=None,
                        help="audio code variant; overrides audio_code.variant")
    parser.add_argument("--layers", type=int, nargs="+", default=None,
                        help="MERT layer indices used for the code; overrides audio_code.layers")
    parser.add_argument("--smoke", action="store_true",
                        help="synthetic codes and features, few steps; checks the code path only")
    return parser


def setup(args):
    config = load_config(args.config)
    if args.seed is not None:
        config["system"]["seed"] = args.seed
    if args.tag:
        config["system"]["tag"] = args.tag
    if args.code:
        config["audio_code"]["variant"] = args.code
    if args.layers:
        config["audio_code"]["layers"] = args.layers
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    return config, device


def run_name(stem, smoke):
    return f"f1_{stem}" + ("_smoke" if smoke else "")


def load_layers(config, smoke=False):
    """All tracks: (ids, array (n_tracks, n_windows, 13, 768), split name per track)."""
    ids, layers = _synthetic_layers() if smoke else load_shards(config["paths"]["mert_embeddings"])
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    splits = np.array([split_of(track_id, val, test) for track_id in ids])
    return np.asarray(ids), layers, splits


def codes_of(layers, splits, variant, whiten_dim, layer_index=None):
    """Codes of every track (n_tracks, n_windows, d) and the transform fitted on training tracks."""
    transform = fit_code_transform(layers[splits == "train"], variant, whiten_dim, layer_index)
    return apply_code_transform(layers, transform), transform


def load_art(config, device, smoke=False):
    """DINOv2 features of every painting: (file names, features (n, 384), split name per painting).

    Features are computed once and cached in paths.dino_art_features.
    """
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    if smoke:
        rng = np.random.default_rng(1)
        centres = rng.normal(size=(6, FEATURE_DIM))
        features = (centres[rng.integers(6, size=400)] + 0.7 * rng.normal(size=(400, FEATURE_DIM))).astype(np.float32)
        names = np.array([f"synthetic_{i:04d}.png" for i in range(400)])
    else:
        cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
        if not cache.exists():
            dataset = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
            dino = DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)
            out = []
            with torch.no_grad():
                for images in DataLoader(dataset, batch_size=64):
                    out.append(dino(images.to(device)).float().cpu())
            cache.parent.mkdir(parents=True, exist_ok=True)
            np.savez(cache, names=np.asarray(dataset.image_files), features=torch.cat(out).numpy())
        with np.load(cache) as data:
            names, features = data["names"], data["features"]
    splits = np.array([split_of(name, val, test) for name in names])
    return names, features, splits


def energy_distance(x, y):
    return 2 * torch.cdist(x, y).mean() - torch.cdist(x, x).mean() - torch.cdist(y, y).mean()


def train_mapper(config, train_codes, val_codes, art_train, art_val, device, variant, steps, seed, log=None):
    """Fit M: audio code -> image feature. Returns (mapper, code_mean, code_std).

    L = L_dist + lambda_struct * L_struct
      L_dist    distribution_term between a batch of outputs and a batch of training-art features,
                both divided by the standard deviation of the training-art features
      L_struct  1 - Pearson(D_a, D_f); D_a, D_f: cosine distances between the audio codes and
                between the outputs of a batch
    """
    hp = config["mapper"]
    use_struct, shuffle = VARIANTS[variant]
    g = torch.Generator().manual_seed(seed)
    torch.manual_seed(seed)
    train_codes = torch.as_tensor(train_codes, device=device)
    val_codes = torch.as_tensor(val_codes, device=device)
    art_train = torch.as_tensor(art_train, device=device)
    art_val = torch.as_tensor(art_val, device=device)
    code_mean = train_codes.mean(dim=(0, 1))
    code_std = train_codes.std(dim=(0, 1)) + 1e-6
    scale = art_train.std()
    mapper = Mapper(train_codes.shape[-1], FEATURE_DIM, hp["hidden"], hp["depth"]).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=hp["lr"])
    batch = min(hp["batch_size"], train_codes.shape[0])

    def sample(codes):
        tracks = torch.randperm(codes.shape[0], generator=g)[:batch].to(device)
        windows = torch.randint(codes.shape[1], (len(tracks),), generator=g).to(device)
        return codes[tracks, windows]

    def losses(codes, real, permute=None):
        out = mapper((codes - code_mean) / code_std)
        if hp["distribution_term"] == "swd":
            l_dist = sliced_wasserstein(out / scale, real / scale, hp["swd_projections"])
        elif hp["distribution_term"] == "energy":
            l_dist = energy_distance(out / scale, real / scale)
        elif hp["distribution_term"] == "none":
            l_dist = torch.zeros((), device=device)
        else:
            raise ValueError(f"unknown distribution term {hp['distribution_term']!r}")
        source = codes[permute] if permute is not None else codes
        l_struct = distance_correlation_loss(pairwise_distances(source), pairwise_distances(out))
        return l_dist, l_struct

    for step in range(steps):
        codes = sample(train_codes)
        real = art_train[torch.randint(art_train.shape[0], (hp["real_batch_size"],), generator=g).to(device)]
        permute = torch.randperm(codes.shape[0], generator=g).to(device) if shuffle else None
        l_dist, l_struct = losses(codes, real, permute)
        loss = l_dist + (hp["lambda_struct"] * l_struct if use_struct else 0.0)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        if log is not None and step % 50 == 0:
            log.log(split="train", step=step, loss=loss, l_dist=l_dist, l_struct=l_struct)
        if log is not None and ((step + 1) % hp["val_every"] == 0 or step + 1 == steps):
            mapper.eval()
            with torch.no_grad():
                real = art_val[torch.randint(art_val.shape[0], (hp["real_batch_size"],), generator=g).to(device)]
                v_dist, v_struct = losses(sample(val_codes), real)
            mapper.train()
            log.log(split="val", step=step + 1, l_dist=v_dist, l_struct=v_struct)
    return mapper.eval(), code_mean, code_std


@torch.no_grad()
def map_codes(mapper, codes, code_mean, code_std, device):
    """Window-averaged codes (n, d) -> output features (n, 384), numpy."""
    codes = torch.as_tensor(codes, device=device)
    return mapper((codes - code_mean) / code_std).cpu().numpy()


def nearest_paintings(outputs, gallery):
    """Index of the gallery feature with the highest cosine similarity to each output."""
    a = outputs / np.linalg.norm(outputs, axis=1, keepdims=True)
    b = gallery / np.linalg.norm(gallery, axis=1, keepdims=True)
    return (a @ b.T).argmax(axis=1)


def correlation(d_a, d_b):
    iu = np.triu_indices(len(d_a), k=1)
    return float(np.corrcoef(d_a[iu], d_b[iu])[0, 1])


def quick_scores(codes, features, test_art, k):
    """Structure and distribution scores without permutation tests."""
    d_audio, d_image = distance_matrix(codes), distance_matrix(features)
    return {"r": correlation(d_audio, d_image), "knn_overlap": knn_overlap(d_audio, d_image, k)[0],
            "frechet_distance": frechet_distance(features, test_art),
            "participation_ratio": participation_ratio(features)}


def full_scores(codes, features, test_art, k, n_permutations, seed):
    d_audio, d_image = distance_matrix(codes), distance_matrix(features)
    r_pearson, p_pearson = mantel(d_audio, d_image, "pearson", n_permutations, seed)
    r_spearman, p_spearman = mantel(d_audio, d_image, "spearman", n_permutations, seed)
    overlap, chance = knn_overlap(d_audio, d_image, k)
    unit = features / np.linalg.norm(features, axis=1, keepdims=True)
    kid_mean, kid_std = kernel_distance(features, test_art, seed=seed)
    iu = np.triu_indices(len(features), k=1)
    it = np.triu_indices(len(test_art), k=1)
    return {
        "structure": {
            "n_tracks": len(codes), "n_permutations": n_permutations,
            "mantel_pearson": r_pearson, "mantel_pearson_p": p_pearson,
            "mantel_spearman": r_spearman, "mantel_spearman_p": p_spearman,
            "knn_k": k, "knn_overlap": overlap, "knn_overlap_chance": chance,
            "trustworthiness": float(trustworthiness(d_audio, unit, n_neighbors=k, metric="precomputed")),
        },
        "image_distribution": {
            "n": len(features), "n_reference": len(test_art),
            "frechet_distance": frechet_distance(features, test_art),
            "kernel_distance": kid_mean, "kernel_distance_std": kid_std,
            "mean_pairwise_distance": float(d_image[iu].mean()),
            "mean_pairwise_distance_test_art": float(distance_matrix(test_art)[it].mean()),
            "participation_ratio": participation_ratio(features),
            "participation_ratio_test_art": participation_ratio(test_art),
            "participation_ratio_audio": participation_ratio(codes),
        },
    }


def load_images(config, names):
    """Paintings as a tensor (n, 3, size, size) in [-1, 1]."""
    size = config["image_size"]
    out = []
    for name in names:
        image = ImageOps.exif_transpose(Image.open(os.path.join(config["paths"]["abstract_art"], str(name))))
        array = np.asarray(image.convert("RGB").resize((size, size), Image.BILINEAR), dtype=np.float32)
        out.append(torch.from_numpy(array).permute(2, 0, 1) / 127.5 - 1)
    return torch.stack(out)


def image_statistics(images):
    """Feature sets that do not depend on DINOv2: colour, coarse layout, radial power spectrum."""
    out = {"colour_mean_std": torch.cat([images.mean((2, 3)), images.std((2, 3))], 1).numpy()}
    q = ((images * 0.5 + 0.5).clamp(0, 0.999) * 4).long()
    index = (q[:, 0] * 16 + q[:, 1] * 4 + q[:, 2]).flatten(1)
    out["colour_histogram"] = torch.stack([torch.bincount(i, minlength=64).float() / i.numel()
                                           for i in index]).numpy()
    out["layout_8x8"] = torch.nn.functional.adaptive_avg_pool2d(images, 8).flatten(1).numpy()
    grey = images.mean(1)
    power = torch.fft.fftshift(torch.fft.fft2(grey - grey.mean((1, 2), keepdim=True)).abs() ** 2, dim=(1, 2))
    side = grey.shape[-1]
    yy, xx = torch.meshgrid(torch.arange(side) - side // 2, torch.arange(side) - side // 2, indexing="ij")
    radius = torch.sqrt((yy ** 2 + xx ** 2).float()).long().clamp(max=side // 2 - 1)
    bins = torch.stack([power[:, radius == k].mean(1) for k in range(1, side // 2)], 1)
    out["power_spectrum"] = torch.log(bins + 1e-8).numpy()
    return out


def _synthetic_layers(n_tracks=600, n_windows=3, dim=768, n_clusters=8, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(n_clusters, dim))
    tracks = centres[rng.integers(n_clusters, size=n_tracks)] + 0.5 * rng.normal(size=(n_tracks, dim))
    codes = tracks[:, None, :] + 0.1 * rng.normal(size=(n_tracks, n_windows, dim))
    scale = rng.uniform(0.2, 5.0, size=(13, 1))
    offset = rng.normal(size=(13, dim))
    layers = codes[:, :, None, :] * scale + offset
    return [f"synthetic_{i:05d}" for i in range(n_tracks)], layers.astype(np.float16)
