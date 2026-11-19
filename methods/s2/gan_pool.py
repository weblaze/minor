"""A pool of G-GAN images as a gallery for S2: generate, drop near-blank images, embed with DINOv2.

Near-blank: the standard deviation of the grey values of an image (pixel values in [-1, 1]) is below
`--min-std`, or below `--grey-std` while its mean saturation (max minus min channel, in [0, 1]) is below
`--min-sat` (pale grey or cream tiles). Plain saturated colour fields are kept. The share of training
paintings that the same rule would drop is reported.

Writes `<out>/pool.npz` (names, features (n, 384), grey std, seed of the kept images) and `<out>/images/`
(PNG per kept image, so the pool can be viewed and referenced by name).

Usage:
  python methods/s2/gan_pool.py --generator <run>/generator.pt --out runs/gan_pool_<name> [--n 6000]
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torchvision.utils import save_image

from abstraction.data.datasets import ImageDataset
from abstraction.data.splits import split_of
from abstraction.generators.gan import GANGenerator
from abstraction.pipeline.features import build_dino
from abstraction.utils.config import load_config

ROOT = Path(__file__).resolve().parents[2]


def grey_std(images):
    return images.mean(dim=1).flatten(1).std(dim=1)


def saturation(images):
    x = images * 0.5 + 0.5
    return (x.max(dim=1).values - x.min(dim=1).values).flatten(1).mean(dim=1)


def blank(images, args):
    s, sat = grey_std(images), saturation(images)
    return (s < args.min_std) | ((s < args.grey_std) & (sat < args.min_sat))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--generator", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n", type=int, default=6000)
    parser.add_argument("--min-std", type=float, default=0.06)
    parser.add_argument("--grey-std", type=float, default=0.10)
    parser.add_argument("--min-sat", type=float, default=0.08)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    config = load_config(ROOT / "methods" / "s2" / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    art = ImageDataset(config["paths"]["abstract_art"], size=config["image_size"], train=False)
    train_idx = [i for i, f in enumerate(art.image_files)
                 if split_of(Path(f).name, config["split"]["val_percent"], config["split"]["test_percent"]) == "train"]
    art_blank = torch.cat([blank(art[i][None], args) for i in train_idx[::5]]).float().mean().item()

    generator = GANGenerator.load(args.generator, config, device)
    dino = build_dino(config, device, False)
    torch.manual_seed(args.seed)
    names, features, stds = [], [], []
    with torch.no_grad():
        for start in range(0, args.n, 100):
            z = torch.randn(min(100, args.n - start), *generator.latent_shape, device=device)
            images = generator.decode(z).clamp(-1, 1)
            s = grey_std(images)
            keep = ~blank(images, args)
            feats = dino(images[keep]).float().cpu()
            for j in torch.nonzero(keep).flatten().tolist():
                name = f"gan_{start + j:05d}.png"
                save_image(images[j] * 0.5 + 0.5, out / "images" / name)
                names.append(name)
            features.append(feats)
            stds.append(s[keep].cpu())
    features = torch.cat(features).numpy()
    np.savez(out / "pool.npz", names=np.array(names), features=features, grey_std=torch.cat(stds).numpy(),
             n_generated=args.n, min_std=args.min_std, grey_std_limit=args.grey_std, min_sat=args.min_sat,
             art_share_blank=art_blank)
    print(f"kept {len(names)} of {args.n}; share of training paintings the same rule calls blank {art_blank:.3f}")


if __name__ == "__main__":
    main()
