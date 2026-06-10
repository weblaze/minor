"""Generate images from audio (or dreams) with the trained latent diffusion model.

Usage:
  python sample.py --audio path/to/song.mp3 [--n 4] [--guidance 3.0] [--steps 50]
  python sample.py --dream [--n 4]
"""
import argparse
from pathlib import Path

import torchvision.utils as vutils

from abstraction.pipelines.latent_diffusion import generate_diffusion, load_models
from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, default=None, help="path to an mp3/wav file")
    parser.add_argument("--dream", action="store_true", help="null-conditioned dream mode")
    parser.add_argument("--n", type=int, default=1, help="number of images to generate")
    parser.add_argument("--guidance", type=float, default=None, help="CFG scale (config default if omitted)")
    parser.add_argument("--steps", type=int, default=None, help="denoising steps (config default if omitted)")
    parser.add_argument("--out", type=str, default=None, help="output directory")
    args = parser.parse_args()

    if not args.audio and not args.dream:
        parser.error("provide --audio <file> or --dream")

    config = load_config(Path(__file__).parent / "config.yaml")
    sampling = config["sampling"]
    guidance = args.guidance if args.guidance is not None else sampling["guidance_scale"]
    steps = args.steps if args.steps is not None else sampling["num_steps"]
    out_dir = Path(args.out) if args.out else Path(config["paths"]["outputs_dir"]) / "02_latent_diffusion_clap"
    out_dir.mkdir(parents=True, exist_ok=True)

    clap, unet, codec, scheduler = load_models(config)

    stem = "dream" if args.dream else Path(args.audio).stem
    for i in range(args.n):
        image = generate_diffusion(
            clap, unet, codec, scheduler,
            audio_path=None if args.dream else args.audio,
            num_steps=steps, guidance_scale=guidance, seed=None,
        )
        out_path = out_dir / f"{stem}_{i:02d}.png"
        vutils.save_image(image, out_path)
        print(f"saved {out_path}")


if __name__ == "__main__":
    main()
