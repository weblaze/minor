"""Extract BYOL-A embeddings for every FMA track, mirroring the clap_features
layout so approach 02's trainer consumes them via cond.source: byol_a.

Embedding = encoder output averaged over consecutive crops of the full track.

Usage:
  python extract_byol_features.py
"""
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from abstraction.utils.checkpoints import load_checkpoint
from abstraction.utils.config import load_config

from byol_a_model import N_MELS, TARGET_FRAMES, AudioEncoder
from train_byol_a import log_mel


def main():
    config = load_config(Path(__file__).parent / "config.yaml")
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    encoder = AudioEncoder(embed_dim=config["cond"]["dim"]).to(device)
    load_checkpoint(encoder, config["paths"]["byol_checkpoint"], device=device)
    encoder.eval()

    fma_dir = config["paths"]["fma_audio"]
    out_dir = config["paths"]["byol_features"]
    cache_dir = config["paths"]["mel_cache"]
    os.makedirs(out_dir, exist_ok=True)

    files = []
    for genre in os.listdir(fma_dir):
        genre_path = os.path.join(fma_dir, genre)
        if os.path.isdir(genre_path):
            files += [os.path.join(genre_path, f) for f in os.listdir(genre_path)
                      if f.endswith((".mp3", ".wav"))]

    for audio_path in tqdm(files, desc="Extracting BYOL-A features"):
        out_path = os.path.join(out_dir, Path(audio_path).stem + "_byol.npy")
        if os.path.exists(out_path):
            continue
        mel = log_mel(audio_path, cache_dir)
        crops = []
        for start in range(0, max(1, mel.shape[1] - TARGET_FRAMES), TARGET_FRAMES):
            crop = mel[:, start:start + TARGET_FRAMES]
            if crop.shape[1] < TARGET_FRAMES:
                crop = np.pad(crop, ((0, 0), (0, TARGET_FRAMES - crop.shape[1])))
            crop = (crop - crop.mean()) / (crop.std() + 1e-8)
            crops.append(torch.from_numpy(crop).unsqueeze(0))
        with torch.no_grad():
            embs = encoder(torch.stack(crops).to(device))
        np.save(out_path, embs.mean(dim=0).cpu().numpy())

    print(f"Done. Features in {out_dir}")


if __name__ == "__main__":
    main()
