"""Turn the unpaired audio/image datasets into trainable (audio, image) pairs.

Strategies:
  random — each CLAP embedding gets a uniformly random image. Baseline: with
           random pairs the UNet learns to ignore conditioning, which is the
           failure mode of the old index-modulo pairing. Train it once to have
           a comparison point.
  pseudo — mood-vocabulary bridge. CLAP and CLIP share no embedding space, but
           both align to text. Each audio clip is scored against mood words via
           CLAP text embeddings, each image against the same words via CLIP.
           Audio is paired with an image whose mood profile is most similar.

Usage:
  python make_pairs.py [--strategy pseudo|random] [--seed 42]
"""
import argparse
import json
import os
import random
from pathlib import Path

import numpy as np

from abstraction.utils.config import load_config

MOODS = [
    "energetic", "calm", "dark", "joyful", "melancholic", "chaotic",
    "dreamy", "tense", "warm", "cold", "aggressive", "gentle",
    "mysterious", "triumphant", "lonely", "playful",
]
AUDIO_TEMPLATE = "this music feels {}"
IMAGE_TEMPLATE = "an abstract painting that feels {}"

IMAGE_EXTS = (".png", ".jpg", ".jpeg")


def mood_profiles_audio(clap_dir, config):
    from abstraction.audio.clap import ClapEncoder

    files = sorted(f for f in os.listdir(clap_dir) if f.endswith(".npy"))
    if not files:
        raise FileNotFoundError(f"No CLAP embeddings in {clap_dir} — run preprocessing/extract_clap_features.py")

    encoder = ClapEncoder(config["paths"]["clap_checkpoint"], device=config["system"]["device"])
    text_embs = encoder.embed_texts([AUDIO_TEMPLATE.format(m) for m in MOODS])
    text_embs = text_embs / np.linalg.norm(text_embs, axis=1, keepdims=True)

    audio_embs = np.stack([np.load(os.path.join(clap_dir, f)) for f in files])
    audio_embs = audio_embs / (np.linalg.norm(audio_embs, axis=1, keepdims=True) + 1e-8)

    profiles = audio_embs @ text_embs.T          # [N_audio, len(MOODS)]
    return files, profiles


def mood_profiles_images(image_dir, config, batch_size=64):
    import open_clip
    import torch
    from PIL import Image

    files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))
    if not files:
        raise FileNotFoundError(f"No images in {image_dir}")

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    model = model.to(device).eval()
    tokenizer = open_clip.get_tokenizer("ViT-B-32")

    with torch.no_grad():
        text_embs = model.encode_text(tokenizer([IMAGE_TEMPLATE.format(m) for m in MOODS]).to(device))
        text_embs = text_embs / text_embs.norm(dim=-1, keepdim=True)

        profiles = []
        for start in range(0, len(files), batch_size):
            batch = files[start:start + batch_size]
            images = torch.stack([
                preprocess(Image.open(os.path.join(image_dir, f)).convert("RGB")) for f in batch
            ]).to(device)
            img_embs = model.encode_image(images)
            img_embs = img_embs / img_embs.norm(dim=-1, keepdim=True)
            profiles.append((img_embs @ text_embs.T).cpu().numpy())
            print(f"  images {min(start + batch_size, len(files))}/{len(files)}", end="\r")
    print()
    return files, np.concatenate(profiles)


def pair_pseudo(config, rng):
    clap_dir = config["paths"]["clap_features"]
    image_dir = config["paths"]["abstract_art"]
    top_k = config["pairing"].get("top_k", 3)

    print("Scoring audio against mood vocabulary (CLAP)...")
    audio_files, audio_profiles = mood_profiles_audio(clap_dir, config)
    print("Scoring images against mood vocabulary (CLIP)...")
    image_files, image_profiles = mood_profiles_images(image_dir, config)

    audio_profiles = audio_profiles / (np.linalg.norm(audio_profiles, axis=1, keepdims=True) + 1e-8)
    image_profiles = image_profiles / (np.linalg.norm(image_profiles, axis=1, keepdims=True) + 1e-8)

    pairs = []
    similarity = audio_profiles @ image_profiles.T   # [N_audio, N_images]
    for i, audio_file in enumerate(audio_files):
        candidates = np.argsort(similarity[i])[-top_k:]
        chosen = int(candidates[rng.randrange(len(candidates))])
        pairs.append({"audio": audio_file, "image": image_files[chosen]})
    return pairs


def pair_random(config, rng):
    clap_dir = config["paths"]["clap_features"]
    image_dir = config["paths"]["abstract_art"]
    audio_files = sorted(f for f in os.listdir(clap_dir) if f.endswith(".npy"))
    image_files = sorted(f for f in os.listdir(image_dir) if f.lower().endswith(IMAGE_EXTS))
    if not audio_files or not image_files:
        raise FileNotFoundError("Missing CLAP embeddings or images — check datasets/README.md")
    return [{"audio": a, "image": rng.choice(image_files)} for a in audio_files]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["pseudo", "random"], default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    config = load_config(Path(__file__).parent / "config.yaml")
    strategy = args.strategy or config["pairing"]["strategy"]
    rng = random.Random(args.seed)

    print(f"Pairing strategy: {strategy}")
    pairs = pair_pseudo(config, rng) if strategy == "pseudo" else pair_random(config, rng)

    out_path = config["paths"]["pairs_file"]
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"strategy": strategy, "seed": args.seed, "pairs": pairs}, f, indent=2)
    print(f"Wrote {len(pairs)} pairs to {out_path}")


if __name__ == "__main__":
    main()
