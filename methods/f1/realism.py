"""F1 experiment grid: mechanisms that place the outputs of M on the art collection.

Each configuration trains one mapping (seed, steps as in config.yaml) and is scored on the test
tracks for the direct output and for the nearest training painting. A configuration is a dict:

  head     free   output of M is the feature vector
           pca    M outputs k coordinates in the principal axes of the training-art features
           soft   M outputs a query q; the output is the softmax(cos(q, g) / tau)-weighted mean of
                  the gallery features g
           st     as soft in the backward pass; the forward pass uses the nearest gallery feature
  dist     list of [term, weight]; terms: swd, critic (hinge loss against a spectral-norm MLP),
           chamfer (mean distance of each output to its nearest gallery feature and of each real
           feature of the batch to its nearest output)
  struct   list of [term, weight]; terms: global (1 - Pearson of distance matrices), local
           (KL divergence between row-wise neighbour distributions), views (global term averaged
           over the single-layer codes in `views`)
  noise    number of standard-normal inputs appended to the code (0: deterministic mapping)
  batch    tracks per batch

Scores per configuration (test tracks, window-averaged codes):
  direct_*     output of the head; painting_*: nearest training painting to that output (to q for
               soft and st)
  r, knn       Pearson correlation of distance matrices; 10-nearest-neighbour overlap
  fd           Frechet distance to test-art features
  displacement mean cosine distance between an output (query) and its nearest painting
  distinct     number of distinct paintings
  pixel_*      r between audio distances and distances of pixel-based feature sets of the paintings
  view_r       mean r of the painting output against the single-layer codes in `views`

Usage:
  python methods/f1/realism.py --seed 0 [--only NAME ...] [--code mean]
"""
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from abstraction.losses.distribution import sliced_wasserstein
from abstraction.losses.structure import distance_correlation_loss, pairwise_distances
from abstraction.metrics.distribution import frechet_distance
from abstraction.metrics.structure import distance_matrix, knn_overlap
from abstraction.models.mapper import Mapper
from abstraction.utils.runlog import RunLog, set_seed

from common import (FEATURE_DIM, base_parser, codes_of, correlation, image_statistics, load_art, load_images,
                    load_layers, run_name, setup)

VIEWS = (0, 3, 6, 9, 12)
SWD = [["swd", 1.0]]
GLOBAL = [["global", 1.0]]
GRID = {
    "swd": {},
    "struct_only": {"dist": []},
    "swd_batch1024": {"batch": 1024},
    "critic_0.3": {"dist": [["critic", 0.3]]},
    "critic_1": {"dist": [["critic", 1.0]]},
    "critic_3": {"dist": [["critic", 3.0]]},
    "chamfer_1": {"dist": [["chamfer", 1.0]]},
    "chamfer_3": {"dist": [["chamfer", 3.0]]},
    "chamfer_10": {"dist": [["chamfer", 10.0]]},
    "critic_chamfer": {"dist": [["critic", 1.0], ["chamfer", 1.0]]},
    "pca32_swd": {"head": "pca", "k": 32},
    "pca64_swd": {"head": "pca", "k": 64},
    "soft_0.05_swd": {"head": "soft", "tau": 0.05},
    "soft_0.02_swd": {"head": "soft", "tau": 0.02},
    "soft_0.02": {"head": "soft", "tau": 0.02, "dist": []},
    "soft_0.01": {"head": "soft", "tau": 0.01, "dist": []},
    "st_0.05_swd": {"head": "st", "tau": 0.05},
    "st_0.05": {"head": "st", "tau": 0.05, "dist": []},
    "st_0.02": {"head": "st", "tau": 0.02, "dist": []},
    "st_0.1": {"head": "st", "tau": 0.1, "dist": []},
    "swd_local": {"struct": [["global", 1.0], ["local", 1.0]]},
    "st_0.05_local": {"head": "st", "tau": 0.05, "dist": [], "struct": [["global", 1.0], ["local", 1.0]]},
    "st_0.05_local_only": {"head": "st", "tau": 0.05, "dist": [], "struct": [["local", 1.0]]},
    "views_swd": {"struct": [["global", 0.5], ["views", 0.5]]},
    "st_0.05_views": {"head": "st", "tau": 0.05, "dist": [], "struct": [["global", 0.5], ["views", 0.5]]},
    "noise16_swd": {"noise": 16},
    "noise16_critic": {"noise": 16, "dist": [["critic", 1.0]]},
}
DEFAULT = {"head": "free", "dist": SWD, "struct": GLOBAL, "noise": 0, "batch": 256, "tau": 0.05, "k": 64}


def neighbour_kl(d_source, d_target, temperature=0.5):
    """KL divergence between row-wise neighbour distributions; distances divided by their mean."""
    n = d_source.shape[0]
    mask = torch.eye(n, dtype=torch.bool, device=d_source.device)

    def log_p(d):
        d = d / d[~mask].mean()
        return F.log_softmax((-d / temperature).masked_fill(mask, float("-inf")), dim=1)

    ls, lt = log_p(d_source), log_p(d_target)
    return (ls.exp() * (ls - lt).masked_fill(mask, 0.0)).sum(1).mean()


def critic_net():
    sn = nn.utils.spectral_norm
    return nn.Sequential(sn(nn.Linear(FEATURE_DIM, 512)), nn.LeakyReLU(0.2), sn(nn.Linear(512, 512)),
                         nn.LeakyReLU(0.2), sn(nn.Linear(512, 1)))


class Head:
    def __init__(self, spec, gallery):
        self.kind, self.tau = spec["head"], spec["tau"]
        self.gallery = gallery
        self.unit = F.normalize(gallery, dim=1)
        self.out_dim = FEATURE_DIM
        if self.kind == "pca":
            mean = gallery.mean(0)
            _, s, v = torch.pca_lowrank(gallery - mean, q=spec["k"], center=False)
            self.mean, self.axes = mean, v * (s / (len(gallery) - 1) ** 0.5)
            self.out_dim = spec["k"]

    def __call__(self, raw):
        """Returns (output feature, query used for retrieval)."""
        if self.kind == "free":
            return raw, raw
        if self.kind == "pca":
            out = self.mean + raw @ self.axes.T
            return out, out
        sim = F.normalize(raw, dim=1) @ self.unit.T
        soft = torch.softmax(sim / self.tau, dim=1) @ self.gallery
        if self.kind == "soft":
            return soft, raw
        hard = self.gallery[sim.argmax(1)]
        return hard + soft - soft.detach(), raw


def fit(spec, config, codes, views, gallery, device, steps, seed):
    hp = config["mapper"]
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    mean, std = codes.mean(dim=(0, 1)), codes.std(dim=(0, 1)) + 1e-6
    scale = gallery.std()
    head = Head(spec, gallery)
    mapper = Mapper(codes.shape[-1] + spec["noise"], head.out_dim, hp["hidden"], hp["depth"]).to(device)
    optimiser = torch.optim.AdamW(mapper.parameters(), lr=hp["lr"])
    use_critic = any(term == "critic" for term, _ in spec["dist"])
    critic = critic_net().to(device) if use_critic else None
    critic_opt = torch.optim.Adam(critic.parameters(), lr=2e-4, betas=(0.5, 0.9)) if use_critic else None
    batch = min(spec["batch"], codes.shape[0])

    def forward(x, noise=None):
        x = (x - mean) / std
        if spec["noise"]:
            noise = torch.zeros(len(x), spec["noise"], device=device) if noise is None else noise
            x = torch.cat([x, noise], dim=1)
        return head(mapper(x))

    for _ in range(steps):
        tracks = torch.randperm(codes.shape[0], generator=g)[:batch].to(device)
        windows = torch.randint(codes.shape[1], (batch,), generator=g).to(device)
        x = codes[tracks, windows]
        noise = torch.randn(batch, spec["noise"], generator=g).to(device) if spec["noise"] else None
        out, _ = forward(x, noise)
        real = gallery[torch.randint(gallery.shape[0], (hp["real_batch_size"],), generator=g).to(device)]
        if use_critic:
            d_loss = F.relu(1 - critic(real / scale)).mean() + F.relu(1 + critic(out.detach() / scale)).mean()
            critic_opt.zero_grad(set_to_none=True)
            d_loss.backward()
            critic_opt.step()
        d_out = pairwise_distances(out)
        loss = torch.zeros((), device=device)
        for term, weight in spec["dist"]:
            if term == "swd":
                loss = loss + weight * sliced_wasserstein(out / scale, real / scale, hp["swd_projections"])
            elif term == "critic":
                loss = loss - weight * critic(out / scale).mean()
            elif term == "chamfer":
                reference = gallery[torch.randint(gallery.shape[0], (2048,), generator=g).to(device)]
                loss = loss + weight * (torch.cdist(out / scale, reference / scale).min(1).values.mean()
                                        + torch.cdist(real / scale, out / scale).min(1).values.mean())
            else:
                raise ValueError(term)
        for term, weight in spec["struct"]:
            if term == "global":
                loss = loss + weight * distance_correlation_loss(pairwise_distances(x), d_out)
            elif term == "local":
                loss = loss + weight * neighbour_kl(pairwise_distances(x), d_out)
            elif term == "views":
                loss = loss + weight * torch.stack([
                    distance_correlation_loss(pairwise_distances(v[tracks, windows]), d_out) for v in views]).mean()
            else:
                raise ValueError(term)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
    mapper.eval()
    return forward


def score(forward, test, test_views, gallery, test_art, pixel_stats, k):
    with torch.no_grad():
        out, query = forward(test)
        sim = F.normalize(query, dim=1) @ F.normalize(gallery, dim=1).T
        best, nearest = sim.max(1)
    out, nearest = out.cpu().numpy(), nearest.cpu().numpy()
    gallery_np, codes = gallery.cpu().numpy(), test.cpu().numpy()
    painting = gallery_np[nearest]
    d_audio, d_direct, d_painting = distance_matrix(codes), distance_matrix(out), distance_matrix(painting)
    row = {"direct_r": correlation(d_audio, d_direct), "direct_knn": knn_overlap(d_audio, d_direct, k)[0],
           "direct_fd": frechet_distance(out, test_art),
           "painting_r": correlation(d_audio, d_painting), "painting_knn": knn_overlap(d_audio, d_painting, k)[0],
           "painting_fd": frechet_distance(painting, test_art),
           "displacement": float(1 - best.mean()), "distinct": int(len(np.unique(nearest)))}
    for name, stats in pixel_stats.items():
        row[f"pixel_{name}"] = correlation(d_audio, distance_matrix(stats[nearest], metric="euclidean"))
    row["view_r"] = float(np.mean([correlation(distance_matrix(v.cpu().numpy()), d_painting) for v in test_views]))
    return row


def gallery_pixel_statistics(config, names):
    cache = Path(config["paths"]["dino_art_features"]) / "pixel_statistics_train.npz"
    if not cache.exists():
        stats = {}
        for start in range(0, len(names), 512):
            for key, value in image_statistics(load_images(config, names[start:start + 512])).items():
                stats.setdefault(key, []).append(value)
        np.savez(cache, names=names, **{key: np.concatenate(value) for key, value in stats.items()})
    with np.load(cache) as data:
        assert (data["names"] == names).all()
        return {key: data[key] for key in data.files if key != "names"}


def main():
    parser = base_parser(__doc__)
    parser.add_argument("--only", nargs="+", default=None)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    config, device = setup(args)
    seed = config["system"]["seed"]
    steps = args.steps or (40 if args.smoke else config["mapper"]["steps"])
    set_seed(seed)
    names = args.only or list(GRID)
    config["run"] = {"steps": steps, "grid": {name: {**DEFAULT, **GRID[name]} for name in names}}
    log = RunLog(config["paths"]["runs_dir"], run_name("realism", args.smoke), config, seed)

    _, layers, splits = load_layers(config, args.smoke)
    art_names, art, art_splits = load_art(config, device, args.smoke)
    gallery = torch.from_numpy(art[art_splits == "train"]).to(device)
    test_art = art[art_splits == "test"]
    pixel_stats = {} if args.smoke else gallery_pixel_statistics(config, art_names[art_splits == "train"])
    code = config["audio_code"]
    codes, _ = codes_of(layers, splits, code["variant"], code["whiten_dim"], code["layers"])
    view_codes = [codes_of(layers, splits, "standardised", code["whiten_dim"], [i])[0] for i in VIEWS]
    to = lambda x: torch.from_numpy(x).to(device)
    train, test = to(codes[splits == "train"]), to(codes[splits == "test"].mean(axis=1))
    train_views = [to(v[splits == "train"]) for v in view_codes]
    test_views = [to(v[splits == "test"].mean(axis=1)) for v in view_codes]

    results = {}
    for name in names:
        spec = {**DEFAULT, **GRID[name]}
        forward = fit(spec, config, train, train_views, gallery, device, steps, seed)
        results[name] = score(forward, test, test_views, gallery, test_art, pixel_stats, config["eval"]["knn_k"])
        log.save_json("results.json", results)
    print(json.dumps({"code": code["variant"], "run": str(log.dir)}))
    keys = ["direct_r", "direct_fd", "painting_r", "painting_knn", "painting_fd", "displacement", "distinct", "view_r"]
    pixel = [key for key in next(iter(results.values())) if key.startswith("pixel_")]
    print("name | " + " | ".join(keys + pixel))
    for name, row in results.items():
        print(name + " | " + " | ".join(f"{row[key]:.3f}" if isinstance(row[key], float) else str(row[key])
                                         for key in keys + pixel))


if __name__ == "__main__":
    main()
