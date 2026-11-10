"""Build the art collection from style folders of a WikiArt copy.

Every directory under --source whose name equals one of --styles is read.
Each image is EXIF-transposed, converted to RGB, resized to size x size
(bilinear, aspect ratio not preserved; the same operation ImageDataset applies)
and written as `<style>__<original stem>.png`. Unreadable files are skipped
and counted. No de-duplication is performed.

Output: the image directory and `manifest.csv` inside it with the columns
file, style, source_file, source_width, source_height.

Usage:
  python scripts/build_art_set.py --source <wikiart root> [--out DIR] [--styles A B ...] [--limit N]
"""
import argparse
import csv
import os
from pathlib import Path

from PIL import Image, ImageOps
from tqdm import tqdm

from abstraction.data.datasets import IMAGE_EXTENSIONS
from abstraction.utils.config import load_config

DEFAULT_STYLES = ("Abstract_Expressionism", "Action_painting", "Color_Field_Painting", "Minimalism")
MAX_DEPTH = 5


def find_style_dirs(source, styles):
    source = Path(source)
    found = {}
    for root, dirs, _ in os.walk(source):
        depth = len(Path(root).relative_to(source).parts)
        if depth >= MAX_DEPTH:
            dirs[:] = []
            continue
        for name in list(dirs):
            if name in styles and name not in found:
                found[name] = Path(root) / name
                dirs.remove(name)
    return found


def main():
    config = load_config()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--source", required=True)
    parser.add_argument("--out", default=config["paths"]["abstract_art"])
    parser.add_argument("--styles", nargs="+", default=list(DEFAULT_STYLES))
    parser.add_argument("--size", type=int, default=config["image_size"])
    parser.add_argument("--limit", type=int, default=None, help="maximum number of images per style")
    args = parser.parse_args()

    style_dirs = find_style_dirs(args.source, set(args.styles))
    missing = sorted(set(args.styles) - set(style_dirs))
    if missing:
        raise SystemExit(f"style directories not found under {args.source}: {missing}")

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = [], 0
    for style in sorted(style_dirs):
        files = sorted(p for p in style_dirs[style].iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        if args.limit is not None:
            files = files[:args.limit]
        for path in tqdm(files, desc=style):
            try:
                with Image.open(path) as raw:
                    image = ImageOps.exif_transpose(raw).convert("RGB")
            except Exception:
                skipped += 1
                continue
            name = f"{style}__{path.stem}.png"
            image.resize((args.size, args.size), Image.BILINEAR).save(out / name)
            rows.append([name, style, path.name, image.width, image.height])

    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["file", "style", "source_file", "source_width", "source_height"])
        writer.writerows(rows)
    counts = {style: sum(1 for r in rows if r[1] == style) for style in sorted(style_dirs)}
    print(f"wrote {len(rows)} images to {out}; skipped {skipped}; per style: {counts}")


if __name__ == "__main__":
    main()
