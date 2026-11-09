"""Extract DINOv2 ViT-S/14 features for every image of the art collection.

Images are resized to `image_size`, then processed exactly as generated images
are during training (see abstraction/encoders/dino.py). Output: one shard with
the file names and an array (n_images, 384), float32.

Usage:
  python scripts/extract_dino.py [--images DIR] [--out DIR]
"""
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from abstraction.data.datasets import ImageDataset
from abstraction.data.embeddings import save_shard
from abstraction.encoders.dino import DinoFeatures
from abstraction.utils.config import load_config


def main():
    config = load_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", default=config["paths"]["abstract_art"])
    parser.add_argument("--out", default=config["paths"]["dino_art_features"])
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    device = torch.device(config["system"]["device"] if torch.cuda.is_available() else "cpu")
    dino = DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)
    dataset = ImageDataset(args.images, size=config["image_size"], train=False)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=2)

    features = []
    with torch.no_grad():
        for images in tqdm(loader):
            features.append(dino(images.to(device)).cpu().numpy())
    save_shard(args.out, 0, dataset.image_files, np.concatenate(features).astype(np.float32))
    print(f"wrote features for {len(dataset)} images")


if __name__ == "__main__":
    main()
