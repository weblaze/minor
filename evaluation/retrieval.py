"""Zero-shot audio→image retrieval: do generated images sit close to their
conditioning audio in embedding space?

Audio i is embedded with CLAP, image i with OpenCLIP; Recall@K measures how
often image i is in the top-K most similar images for audio i. Above-random
recall means the generations carry audio-specific signal.

Requires: pip install open_clip_torch

Usage:
  python evaluation/retrieval.py --audio_dir <songs> --image_dir <generated images>
  (files are paired by sorted order: audio[i] <-> image[i])
"""
import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm

from abstraction.audio.clap import ClapEncoder
from abstraction.utils.config import load_config


def compute_recall_at_k(similarity_matrix, k=1):
    n = similarity_matrix.shape[0]
    sorted_indices = torch.argsort(similarity_matrix, descending=True, dim=1)
    hits = sum(1 for i in range(n) if i in sorted_indices[i, :k])
    return hits / n


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio_dir", type=str, required=True, help="conditioning source audio files")
    parser.add_argument("--image_dir", type=str, required=True, help="generated images, paired by sorted order")
    parser.add_argument("--out", type=str, default="evaluation/results/retrieval.json")
    args = parser.parse_args()

    import open_clip

    config = load_config()
    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    print(f"Running evaluation on: {device}")

    print("Loading OpenCLIP for image embeddings...")
    clip_model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained="laion2b_s34b_b79k")
    clip_model = clip_model.to(device).eval()

    print("Loading LAION-CLAP for audio embeddings...")
    clap = ClapEncoder(config["paths"]["clap_checkpoint"], device=device)

    audio_files = sorted(list(Path(args.audio_dir).glob("*.mp3")) + list(Path(args.audio_dir).glob("*.wav")))
    image_files = sorted(list(Path(args.image_dir).glob("*.png")) + list(Path(args.image_dir).glob("*.jpg")))
    n = min(len(audio_files), len(image_files))
    if n == 0:
        print("ERROR: no audio/image files found in the provided directories.")
        return
    audio_files, image_files = audio_files[:n], image_files[:n]
    print(f"Evaluating {n} audio-image pairs.")

    print("Extracting audio embeddings...")
    audio_embs = []
    for audio_path in tqdm(audio_files):
        emb = clap.embed_file(audio_path)
        audio_embs.append(emb / emb.norm(dim=-1, keepdim=True))

    print("Extracting image embeddings...")
    image_embs = []
    with torch.no_grad():
        for img_path in tqdm(image_files):
            img = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)
            emb = clip_model.encode_image(img)
            image_embs.append(emb / emb.norm(dim=-1, keepdim=True))

    similarity = torch.cat(audio_embs) @ torch.cat(image_embs).t()

    results = {"n": n}
    for k in (1, 5, 10):
        if n >= k:
            results[f"recall@{k}"] = compute_recall_at_k(similarity, k=k)
            results[f"random_baseline_r{k}"] = k / n

    print("\n" + "=" * 40)
    print("Retrieval Evaluation Results")
    print("=" * 40)
    for k in (1, 5, 10):
        if f"recall@{k}" in results:
            r, base = results[f"recall@{k}"], results[f"random_baseline_r{k}"]
            print(f"Recall@{k}: {r:.4f}  (random {base:.4f}, {r / base:.2f}x)")
    print("=" * 40)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
