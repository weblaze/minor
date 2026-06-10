"""Music video generation: one diffusion keyframe per audio segment, slerp
interpolation between adjacent keyframe latents, ffmpeg assembly with the
original audio muxed back in.

Temporal coherence comes from sharing the SAME initial noise tensor across all
segments — adjacent segments then differ only by their conditioning, so the
visuals evolve with the music instead of jumping randomly. Optional dream
segments (null conditioning) let the model "wander" between audio cues.

Requires: pip install imageio[ffmpeg]   (or pip install -e .[video])

Usage:
  python generate_video.py --audio song.mp3 [--out out.mp4] [--dream-every 4]
"""
import argparse
import math
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import torch

from abstraction.pipelines.latent_diffusion import load_models
from abstraction.utils.config import load_config


def slerp(a, b, t):
    """Spherical interpolation between two latent tensors."""
    a_flat, b_flat = a.flatten(), b.flatten()
    omega = torch.acos(torch.clamp(
        torch.dot(a_flat / a_flat.norm(), b_flat / b_flat.norm()), -1, 1))
    if omega.abs() < 1e-4:
        return (1 - t) * a + t * b
    so = torch.sin(omega)
    return (torch.sin((1 - t) * omega) / so) * a + (torch.sin(t * omega) / so) * b


@torch.no_grad()
def denoise(unet, scheduler, init_noise, conditioning, null_cond, num_steps, guidance_scale):
    latents = init_noise.clone()
    timesteps = np.linspace(scheduler.num_train_timesteps - 1, 0, num_steps).astype(int)
    use_cfg = guidance_scale > 1.0 and not torch.equal(conditioning, null_cond)
    for t in timesteps:
        t_tensor = torch.tensor([t], device=latents.device).long()
        noise_pred = unet(latents, t_tensor, conditioning)
        if use_cfg:
            noise_null = unet(latents, t_tensor, null_cond)
            noise_pred = noise_null + guidance_scale * (noise_pred - noise_null)
        latents = scheduler.step(noise_pred, t, latents)
    return latents


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", type=str, required=True)
    parser.add_argument("--out", type=str, default=None)
    parser.add_argument("--dream-every", type=int, default=None,
                        help="every Nth segment is a dream (overrides config)")
    args = parser.parse_args()

    import imageio.v2 as imageio
    import librosa

    config = load_config(Path(__file__).parent / "config.yaml")
    video_cfg = config["video"]
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    torch.manual_seed(config["system"]["seed"])

    clap, unet, codec, scheduler = load_models(config)
    null_cond = torch.zeros((1, config["cond"]["dim"]), device=device)
    dream_every = args.dream_every if args.dream_every is not None else video_cfg["dream_every"]

    print(f"Loading audio: {args.audio}")
    y, sr = librosa.load(args.audio, sr=48000)
    seg_samples = video_cfg["segment_seconds"] * sr
    n_segments = max(1, math.ceil(len(y) / seg_samples))
    print(f"{n_segments} segments of {video_cfg['segment_seconds']}s")

    # shared initial noise = temporal coherence
    init_noise = torch.randn(
        (1, codec.latent_channels, codec.spatial_size, codec.spatial_size), device=device)

    keyframes = []
    for i in range(n_segments):
        segment = y[i * seg_samples:(i + 1) * seg_samples]
        if dream_every and (i + 1) % dream_every == 0:
            print(f"segment {i + 1}/{n_segments}: dream")
            cond = null_cond
        else:
            print(f"segment {i + 1}/{n_segments}: conditioned")
            cond = clap.embed_array(segment, sr)
        keyframes.append(denoise(unet, scheduler, init_noise, cond, null_cond,
                                 video_cfg["num_steps"], video_cfg["guidance_scale"]))

    print("Interpolating and decoding frames...")
    frames = []
    n_trans = video_cfg["frames_per_transition"]
    for i in range(len(keyframes) - 1):
        for j in range(n_trans):
            latent = slerp(keyframes[i], keyframes[i + 1], j / n_trans)
            image = codec.decode(latent)
            image = ((image + 1) / 2).clamp(0, 1)
            frame = (image.squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
            frames.append(frame)
    image = codec.decode(keyframes[-1])
    frames.append((((image + 1) / 2).clamp(0, 1)
                   .squeeze(0).permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8))

    out_path = Path(args.out) if args.out else \
        Path(config["paths"]["outputs_dir"]) / "04_temporal_video" / (Path(args.audio).stem + ".mp4")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        silent = Path(tmp) / "silent.mp4"
        imageio.mimwrite(silent, frames, fps=video_cfg["fps"], quality=8)

        from imageio_ffmpeg import get_ffmpeg_exe
        subprocess.run([
            get_ffmpeg_exe(), "-y",
            "-i", str(silent), "-i", args.audio,
            "-c:v", "copy", "-c:a", "aac", "-shortest",
            str(out_path),
        ], check=True, capture_output=True)

    print(f"Music video saved to {out_path} "
          f"({len(frames)} frames @ {video_cfg['fps']} fps)")


if __name__ == "__main__":
    main()
