"""The ablation: does CLAP (semantic supervision) or BYOL-A (pure acoustics)
produce better music representations for this project?

Two probes over FMA-small, using genre folders as labels:
  1. linear-probe genre accuracy (logistic regression on frozen embeddings)
  2. t-SNE colored by genre, side by side

Usage:
  python eval_encoder.py
"""
import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split

from abstraction.utils.config import load_config


def collect(features_dir, suffix, genre_of):
    """Load embeddings whose track stem appears in the genre map."""
    embs, labels = [], []
    for f in sorted(os.listdir(features_dir)):
        if not f.endswith(".npy"):
            continue
        stem = f[: -len(suffix)] if f.endswith(suffix) else Path(f).stem
        if stem in genre_of:
            embs.append(np.load(os.path.join(features_dir, f)))
            labels.append(genre_of[stem])
    return np.stack(embs), np.array(labels)


def genre_map(fma_dir):
    mapping = {}
    for genre in os.listdir(fma_dir):
        genre_path = os.path.join(fma_dir, genre)
        if os.path.isdir(genre_path):
            for f in os.listdir(genre_path):
                if f.endswith((".mp3", ".wav")):
                    mapping[f.rsplit(".", 1)[0]] = genre
    return mapping


def linear_probe(embs, labels, seed):
    x_train, x_test, y_train, y_test = train_test_split(
        embs, labels, test_size=0.25, random_state=seed, stratify=labels)
    clf = LogisticRegression(max_iter=2000).fit(x_train, y_train)
    return clf.score(x_test, y_test)


def main():
    config = load_config(Path(__file__).parent / "config.yaml")
    seed = config["system"]["seed"]
    genre_of = genre_map(config["paths"]["fma_audio"])
    genres = sorted(set(genre_of.values()))
    print(f"{len(genre_of)} labeled tracks across {len(genres)} genres")

    encoders = {}
    clap_dir = config["paths"]["clap_features"]
    if os.path.isdir(clap_dir) and os.listdir(clap_dir):
        encoders["CLAP"] = collect(clap_dir, "_clap.npy", genre_of)
    byol_dir = config["paths"]["byol_features"]
    if os.path.isdir(byol_dir) and os.listdir(byol_dir):
        encoders["BYOL-A"] = collect(byol_dir, "_byol.npy", genre_of)
    if not encoders:
        print("No features found — run preprocessing/extract_clap_features.py and extract_byol_features.py first.")
        return

    results = {}
    fig, axes = plt.subplots(1, len(encoders), figsize=(9 * len(encoders), 8))
    if len(encoders) == 1:
        axes = [axes]

    for ax, (name, (embs, labels)) in zip(axes, encoders.items()):
        acc = linear_probe(embs, labels, seed)
        results[name] = {"linear_probe_genre_acc": acc, "n": len(labels)}
        print(f"{name}: linear-probe genre accuracy {acc:.3f} ({len(labels)} tracks)")

        coords = TSNE(n_components=2, random_state=seed,
                      perplexity=min(30, len(embs) - 1)).fit_transform(embs)
        for genre in genres:
            mask = labels == genre
            ax.scatter(coords[mask, 0], coords[mask, 1], label=genre, alpha=0.6, s=12)
        ax.set_title(f"{name} — probe acc {acc:.3f}")
        ax.legend(fontsize=7)

    out_dir = Path("evaluation/results")
    out_dir.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(out_dir / "encoder_ablation_tsne.png", dpi=150)
    with open(out_dir / "encoder_ablation.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Wrote {out_dir / 'encoder_ablation.json'} and the t-SNE plot.")


if __name__ == "__main__":
    main()
