"""LPIPS diversity: mean pairwise perceptual distance among generations.

Two readings:
  --image_dir with N generations from the SAME audio  -> within-song diversity.
    Near-zero = mode collapse (the model ignores the noise seed).
  --image_dir with one generation per DIFFERENT audio -> across-song diversity.
    Near-zero = the model ignores conditioning and always draws the same thing.

Requires: pip install lpips

Usage:
  python evaluation/lpips_diversity.py --image_dir <dir> [--max-images 32]
"""
import argparse
import itertools
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--max-images", type=int, default=32, help="cap pairwise comparisons")
    parser.add_argument("--out", type=str, default="evaluation/results/lpips_diversity.json")
    args = parser.parse_args()

    import lpips

    config = load_config()
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    image_files = sorted(list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg")))
    image_files = image_files[:args.max_images]
    if len(image_files) < 2:
        print("ERROR: need at least 2 images.")
        return
    print(f"Computing pairwise LPIPS over {len(image_files)} images "
          f"({len(image_files) * (len(image_files) - 1) // 2} pairs)")

    loss_fn = lpips.LPIPS(net="alex").to(device)
    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),  # lpips expects [-1, 1]
    ])
    images = [transform(Image.open(f).convert("RGB")).unsqueeze(0).to(device) for f in image_files]

    distances = []
    with torch.no_grad():
        for a, b in tqdm(list(itertools.combinations(range(len(images)), 2))):
            distances.append(loss_fn(images[a], images[b]).item())

    distances = torch.tensor(distances)
    results = {
        "n_images": len(images),
        "lpips_mean": float(distances.mean()),
        "lpips_std": float(distances.std()),
        "lpips_min": float(distances.min()),
    }
    print(f"\nLPIPS diversity: {results['lpips_mean']:.4f} ± {results['lpips_std']:.4f} "
          f"(min {results['lpips_min']:.4f})")
    print("Rule of thumb: < 0.1 mean suggests mode collapse for this image size.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
