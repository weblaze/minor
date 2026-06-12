"""Generate the genre gallery that validates conditioning works.

For each input song: CLAP-embed it, sample several latents from the prior,
decode each with the song's conditioning. If conditioning works, rows differ
systematically by song (palette/texture), columns show within-song variation.

Usage:
  python sample_gallery.py --audio song1.mp3 song2.mp3 song3.mp3 ...
  python sample_gallery.py --audio-dir path/to/folder   # every mp3/wav inside
"""
import argparse
from pathlib import Path

import torch
import torchvision.utils as vutils

from abstraction.audio.clap import ClapEncoder
from abstraction.models.image_vae import ConditionedImageVAE
from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", nargs="*", default=[], help="audio files, one row each")
    parser.add_argument("--audio-dir", type=str, default=None, help="folder of audio files")
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    audio_files = [Path(a) for a in args.audio]
    if args.audio_dir:
        audio_files += sorted(p for p in Path(args.audio_dir).iterdir()
                              if p.suffix.lower() in (".mp3", ".wav"))
    if not audio_files:
        parser.error("provide --audio files or --audio-dir")

    config = load_config(Path(__file__).parent / "config.yaml")
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    n_samples = config["gallery"]["samples_per_song"]

    model = ConditionedImageVAE(latent_channels=config["image_vae"]["latent_channels"]).to(device)
    load_checkpoint(model, config["paths"]["conditioned_checkpoint"], device=device)
    model.eval()

    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)

    torch.manual_seed(config["system"]["seed"])
    # same latents for every song: differences between rows are purely the conditioning
    raw_noise = torch.randn(n_samples, config["image_vae"]["latent_channels"], 16, 16, device=device)

    # Apply spatial smoothing (Gaussian filter) to create spatial correlation (natural image statistics)
    import torch.nn.functional as F
    kernel_size = 5
    sigma = 1.5
    x = torch.arange(-kernel_size//2 + 1., kernel_size//2 + 1.)
    x_grid, y_grid = torch.meshgrid(x, x, indexing='ij')
    kernel = torch.exp(-(x_grid**2 + y_grid**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    kernel = kernel.view(1, 1, kernel_size, kernel_size).repeat(config["image_vae"]["latent_channels"], 1, 1, 1).to(device)
    
    latents = F.conv2d(raw_noise, kernel, padding=kernel_size//2, groups=config["image_vae"]["latent_channels"])
    latents = (latents / latents.std()) * 0.5  # Scale standard deviation to 0.5 so style is guided by conditioning

    rows = []
    for audio in audio_files:
        print(f"embedding {audio.name}...")
        cond = clap.embed_file(audio).repeat(n_samples, 1)
        with torch.no_grad():
            images = model.decode(latents, cond=cond)
        rows.append((images + 1) / 2)

    grid = vutils.make_grid(torch.cat(rows).clamp(0, 1).cpu(), nrow=n_samples)
    out_path = Path(args.out) if args.out else \
        Path(config["paths"]["outputs_dir"]) / "01_conditioned_autoencoder" / "genre_gallery.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    vutils.save_image(grid, out_path)

    print(f"\nGallery saved to {out_path}")
    print("Rows (top to bottom):")
    for audio in audio_files:
        print(f"  {audio.name}")
    print("\nGate check: rows should differ visibly by song. If all rows look "
          "identical, conditioning is not working — revisit finetune.")


if __name__ == "__main__":
    main()
