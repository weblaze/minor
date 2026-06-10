"""FID between the training image distribution and a directory of generations.

Lower is better. Compares distributions, not pairs — measures whether the
generations look like the dataset, not whether they match the audio.

Requires: pip install clean-fid

Usage:
  python evaluation/fid.py --generated <dir> [--reference <dir>]
"""
import argparse
import json
from pathlib import Path

from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generated", type=str, required=True, help="directory of generated images")
    parser.add_argument("--reference", type=str, default=None,
                        help="reference image directory (default: datasets/abstract_art)")
    parser.add_argument("--out", type=str, default="evaluation/results/fid.json")
    args = parser.parse_args()

    from cleanfid import fid

    config = load_config()
    reference = args.reference or config["paths"]["abstract_art"]

    print(f"Computing FID: {args.generated} vs {reference}")
    score = fid.compute_fid(args.generated, reference)
    print(f"FID: {score:.2f}")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"fid": score, "generated": args.generated, "reference": reference}, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
