"""Extract 512-d LAION CLAP embeddings for every track in the FMA dataset."""
import os

import numpy as np
from tqdm import tqdm

from abstraction.audio.clap import ClapEncoder
from abstraction.utils.config import load_config


def process_audio_dataset(encoder, dataset_path, output_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Audio dataset path does not exist: {dataset_path}")
    os.makedirs(output_dir, exist_ok=True)

    for genre_folder in tqdm(os.listdir(dataset_path), desc="Processing Audio Genres"):
        genre_path = os.path.join(dataset_path, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        files = [f for f in os.listdir(genre_path) if f.endswith((".mp3", ".wav"))]
        for file in tqdm(files, desc=f"Processing {genre_folder}", leave=False):
            out_path = os.path.join(output_dir, file.rsplit(".", 1)[0] + "_clap.npy")
            if os.path.exists(out_path):
                continue
            try:
                embedding = encoder.embed_files([os.path.join(genre_path, file)])[0]
                np.save(out_path, embedding)
            except Exception as e:
                print(f"Error processing {file}: {e}")


if __name__ == "__main__":
    config = load_config()
    print("Loading LAION CLAP (music_audioset)...")
    encoder = ClapEncoder(config["paths"]["clap_checkpoint"], device=config["system"]["device"])
    print("Extracting CLAP embeddings...")
    process_audio_dataset(encoder, config["paths"]["fma_audio"], config["paths"]["clap_features"])
    print("CLAP feature extraction complete.")
