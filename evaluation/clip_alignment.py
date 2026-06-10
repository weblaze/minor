"""Mean audio-image alignment score over generated pairs — the headline
"does conditioning work" metric.

Both modalities are projected onto the shared mood vocabulary (CLAP text for
audio, CLIP text for images — the same bridge make_pairs.py uses) and scored
by cosine similarity of their mood profiles. Compare runs (pseudo vs random
pairing, guidance scales): higher = generations track the music better.

Requires: pip install open_clip_torch

Usage:
  python evaluation/clip_alignment.py --audio_dir <songs> --image_dir <generated>
  (files are paired by sorted order: audio[i] <-> image[i])
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from abstraction.audio.clap import ClapEncoder
from abstraction.utils.config import load_config

MOODS = [
    "energetic", "calm", "dark", "joyful", "melancholic", "chaotic",
    "dreamy", "tense", "warm", "cold", "aggressive", "gentle",
    "mysterious", "triumphant", "lonely", "playful",
]
AUDIO_TEMPLATE = "this music feels {}"
IMAGE_TEMPLATE = "an abstract painting that feels {}"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=str, required=True)
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--out", type=str, default="evaluation/results/clip_alignment.json")
    args = parser.parse_args()

    import open_clip

    config = load_config()
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")

    audio_files = sorted(list(Path(args.audio_dir).glob("*.mp3")) + list(Path(args.audio_dir).glob("*.wav")))
    image_files = sorted(list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg")))
    n = min(len(audio_files), len(image_files))
    if n == 0:
        print("ERROR: no audio/image files found.")
        return
    audio_files, image_files = audio_files[:n], image_files[:n]
    print(f"Scoring {n} audio-image pairs.")

    print("Audio mood profiles (CLAP)...")
    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)
    text_embs = clap.embed_texts([AUDIO_TEMPLATE.format(m) for m in MOODS])
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)
    audio_profiles = []
    for audio in tqdm(audio_files):
        emb = clap.embed_file(audio).cpu().numpy()[0]
        emb = emb / (np.linalg.norm(emb) + 1e-8)
        audio_profiles.append(emb @ text_embs.T)
    audio_profiles = np.stack(audio_profiles)

    print("Image mood profiles (CLIP)...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")
    with torch.no_grad():
        clip_text = clip_model.encode_text(tokenizer([IMAGE_TEMPLATE.format(m) for m in MOODS]).to(device))
        clip_text = (clip_text / clip_text.norm(dim=-1, keepdim=True)).cpu().numpy()
        image_profiles = []
        for img_path in tqdm(image_files):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            emb = clip_model.encode_image(img)
            emb = (emb / emb.norm(dim=-1, keepdim=True)).cpu().numpy()[0]
            image_profiles.append(emb @ clip_text.T)
    image_profiles = np.stack(image_profiles)

    audio_profiles = audio_profiles / (np.linalg.norm(audio_profiles, axis=1, keepdims=True) + 1e-8)
    image_profiles = image_profiles / (np.linalg.norm(image_profiles, axis=1, keepdims=True) + 1e-8)

    paired = np.sum(audio_profiles * image_profiles, axis=1)          # matched pairs
    shuffled = audio_profiles @ image_profiles.T                       # all combinations
    random_mean = (shuffled.sum() - paired.sum()) / (n * n - n) if n > 1 else 0.0

    results = {
        "n": n,
        "alignment_mean": float(paired.mean()),
        "alignment_std": float(paired.std()),
        "shuffled_baseline_mean": float(random_mean),
        "gap": float(paired.mean() - random_mean),
    }
    print("\n" + "=" * 40)
    print(f"Alignment (matched pairs): {results['alignment_mean']:.4f} ± {results['alignment_std']:.4f}")
    print(f"Shuffled baseline:         {results['shuffled_baseline_mean']:.4f}")
    print(f"Gap (signal):              {results['gap']:+.4f}")
    print("=" * 40)
    print("Gap > 0 means generations carry audio-specific mood signal.")

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
