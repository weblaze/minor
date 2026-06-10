"""Extract MFCC/spectral/RMS/tempo features for every track in the FMA dataset."""
import os

import numpy as np
from tqdm import tqdm

from abstraction.audio.features import extract_features
from abstraction.utils.config import load_config


def process_audio_dataset(dataset_path, output_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Audio dataset path does not exist: {dataset_path}")
    os.makedirs(output_dir, exist_ok=True)

    for genre_folder in tqdm(os.listdir(dataset_path), desc="Processing Audio"):
        genre_path = os.path.join(dataset_path, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if not file.endswith((".mp3", ".wav")):
                continue
            out_path = os.path.join(output_dir, file.rsplit(".", 1)[0] + ".npy")
            if os.path.exists(out_path):
                continue
            features = extract_features(os.path.join(genre_path, file))
            if features is not None:
                np.save(out_path, features)


if __name__ == "__main__":
    config = load_config()
    print("Extracting audio features...")
    process_audio_dataset(config["paths"]["fma_audio"], config["paths"]["audio_features"])
    print("Audio feature extraction complete.")
