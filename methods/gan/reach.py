"""Reach of the G-GAN: how close can it draw to a given DINOv2 description?

Step 1 of passing a picker's description (F1, S2) to a generator. Targets: DINOv2 features of held-out
(test-split) paintings, which the GAN never saw. For each target the latent z is searched directly:

  init      best of `--pool` prior samples by DINOv2 cosine to the target
  search    Adam on z, `--steps` steps, loss -cos(DINOv2(G(z)), target) + `--prior-weight` * (|z|^2 / d - 1)^2
            (the second term keeps z at the radius of the prior); the best z of the search is kept (checked
            every 20 steps; for `aug` on one augmented draw). Two variants: plain, and `aug`, where the
            image passes through the S3 training augmentation (crop, noise, blur) before DINOv2 at every step,
            so that the match cannot rest on pixel patterns that only DINOv2 reads.

Scores per target (cosine similarity, higher is closer):
  dinov2     to the target's DINOv2 features (the quantity searched)
  dino_v1    DINO v1 (ViT-B/16 CLS, never used in the search) of the drawn image against DINO v1 of the target
             painting: does the match hold for an encoder the search never saw?
  novelty    DINOv2 cosine of the drawn image to its nearest training painting (does it draw the target, or
             fall back on a painting it learnt?)
References: random prior sample; the best of the pool (a picker over GAN samples); the nearest training
painting (a picker over real paintings, as F1 and S2).

Writes results.json and reach_grid.png (rows: target, nearest training painting, best of pool, plain, aug).

Usage:
  python methods/gan/reach.py --vae <gan run>/generator.pt [--n-targets 192] [--steps 300]
"""
from pathlib import Path
import sys

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.utils import save_image

from abstraction.data.datasets import ImageDataset
from abstraction.data.splits import split_of
from abstraction.generators import add_generator_argument, load_generator
from abstraction.pipeline import cli
from abstraction.pipeline.features import build_dino
from abstraction.utils.runlog import RunLog, set_seed

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "s3"))
from common import augment  # noqa: E402
from second_reader import DinoV1  # noqa: E402

CONFIG_PATH = Path(__file__).with_name("config.yaml")
AUG = {"crop_scale": [0.85, 1.0], "noise_std": 0.02, "blur_sigma": 0.6, "blur_prob": 0.3}


def cos(a, b):
    return F.cosine_similarity(a.float(), b.float(), dim=-1)


def main():
    parser = add_generator_argument(cli.base_parser(__doc__, CONFIG_PATH), "gan")
    parser.add_argument("--vae", required=True, help="generator checkpoint")
    parser.add_argument("--n-targets", type=int, default=192)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--pool", type=int, default=512)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--prior-weight", type=float, default=0.1)
    args = parser.parse_args()
    config, device = cli.setup(args)
    seed = config["system"]["seed"]
    set_seed(seed)
    config["run"] = {k: v for k, v in vars(args).items() if k not in ("config",)}
    log = RunLog(config["paths"]["runs_dir"], cli.run_name("gan", "reach", args.smoke), config, seed)

    generator = load_generator(args.generator, args.vae, config, device)
    dino = build_dino(config, device, args.smoke)
    dino_v1 = DinoV1(device)
    for p in list(dino.parameters()) + list(dino_v1.model.parameters()):
        p.requires_grad_(False)
    val, test = config["split"]["val_percent"], config["split"]["test_percent"]
    cache = Path(config["paths"]["dino_art_features"]) / f"cls_{config['dino']['size']}.npz"
    with np.load(cache) as data:
        names, features = data["names"], torch.from_numpy(data["features"]).float().to(device)
    splits = np.array([split_of(n, val, test) for n in names])
    train_features = F.normalize(features[splits == "train"], dim=1)
    train_names = names[splits == "train"]
    rng = np.random.default_rng(seed)
    test_index = rng.choice(np.flatnonzero(splits == "test"), size=args.n_targets, replace=False)
    art = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    row_of = {name: i for i, name in enumerate(art.image_files)}
    load = lambda ns: torch.stack([art[row_of[str(n)]] for n in ns]).to(device)
    g = torch.Generator().manual_seed(seed)
    d = generator.latent_shape[0]

    def draw(z):
        return generator.decode(z * generator.latent_scale)

    @torch.no_grad()
    def score(images, target, target_v1):
        f = dino(images).float()
        return {"dinov2": cos(f, target), "dino_v1": cos(dino_v1(images), target_v1),
                "novelty": (F.normalize(f, dim=1) @ train_features.T).max(dim=1).values}

    rows = {k: {"dinov2": [], "dino_v1": [], "novelty": []} for k in ("random", "best_of_pool", "plain", "aug")}
    nearest = {"dinov2": [], "dino_v1": []}
    grid = []
    for start in range(0, args.n_targets, args.batch):
        index = test_index[start:start + args.batch]
        target = features[index]
        with torch.no_grad():
            target_images = load(names[index])
            target_v1 = dino_v1(target_images).float()
            sims = F.normalize(target, dim=1) @ train_features.T
            near_names = train_names[sims.argmax(dim=1).cpu().numpy()]
            near_images = load(near_names)
            nearest["dinov2"].append(sims.max(dim=1).values)
            nearest["dino_v1"].append(cos(dino_v1(near_images), target_v1))
            pool = generator.prior_sample(args.pool, g).flatten(1).to(device) / generator.latent_scale
            pool_f = torch.cat([dino(draw(pool[i:i + 64])).float() for i in range(0, args.pool, 64)])
            best = (F.normalize(target, dim=1) @ F.normalize(pool_f, dim=1).T).argmax(dim=1)
            z0 = pool[best]
            random_images = draw(pool[:len(index)])
            pool_images = draw(z0)
        for name, images in (("random", random_images), ("best_of_pool", pool_images)):
            for k, v in score(images, target, target_v1).items():
                rows[name][k].append(v)
        found = {}
        for variant in ("plain", "aug"):
            z = z0.clone().requires_grad_(True)
            optimiser = torch.optim.Adam([z], lr=args.lr)
            best_z = z0.clone()
            with torch.no_grad():
                best_s = cos(dino(draw(z0)), target)
            for step in range(args.steps):
                images = draw(z)
                if variant == "aug":
                    images = augment(images, AUG, g)
                loss = -cos(dino(images), target).mean() + args.prior_weight * ((z.pow(2).sum(1) / d - 1) ** 2).mean()
                optimiser.zero_grad(set_to_none=True)
                loss.backward()
                optimiser.step()
                if (step + 1) % 20 == 0 or step + 1 == args.steps:
                    with torch.no_grad():
                        s = cos(dino(augment(draw(z), AUG, g) if variant == "aug" else draw(z)), target)
                        better = s > best_s
                        best_z[better], best_s[better] = z.detach()[better], s[better]
            with torch.no_grad():
                found[variant] = draw(best_z)
            for k, v in score(found[variant], target, target_v1).items():
                rows[variant][k].append(v)
        if start == 0:
            grid = [target_images, near_images, pool_images, found["plain"], found["aug"]]
        print(f"{start + len(index)}/{args.n_targets}: " + " ".join(
            f"{k} {torch.cat(rows[k]['dinov2']).mean():.3f}/{torch.cat(rows[k]['dino_v1']).mean():.3f}" for k in rows)
            + f" nearest {torch.cat(nearest['dinov2']).mean():.3f}/{torch.cat(nearest['dino_v1']).mean():.3f}", flush=True)

    summary = lambda v: {"mean": float(v.mean()), "median": float(v.median()), "q10": float(v.quantile(0.1)),
                         "q90": float(v.quantile(0.9))}
    results = {"n_targets": args.n_targets,
               "nearest_training_painting": {k: summary(torch.cat(v)) for k, v in nearest.items()}}
    for name, block in rows.items():
        results[name] = {k: summary(torch.cat(v)) for k, v in block.items()}
    plain, near = torch.cat(rows["plain"]["dinov2"]), torch.cat(nearest["dinov2"])
    results["share_plain_closer_than_nearest_painting"] = float((plain > near).float().mean())
    results["share_aug_closer_than_nearest_painting"] = float((torch.cat(rows["aug"]["dinov2"]) > near).float().mean())
    log.save_json("results.json", results)
    save_image(torch.cat(grid) * 0.5 + 0.5, log.dir / "reach_grid.png", nrow=args.batch)
    print(f"run directory: {log.dir}")


if __name__ == "__main__":
    main()
