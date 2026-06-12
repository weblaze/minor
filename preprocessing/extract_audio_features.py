"""Extract MFCC/spectral/RMS/tempo features for every track in the FMA dataset."""
import os

import numpy as np
from tqdm import tqdm

from abstraction.audio.features import extract_features
from abstraction.utils.config import load_config


def process_single_file(args):
    file_path, out_path = args
    if os.path.exists(out_path):
        return
    try:
        features = extract_features(file_path)
        if features is not None:
            np.save(out_path, features)
    except Exception as e:
        print(f"Error processing {file_path}: {e}")


def process_audio_dataset(dataset_path, output_dir):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Audio dataset path does not exist: {dataset_path}")
    os.makedirs(output_dir, exist_ok=True)

    tasks = []
    for genre_folder in os.listdir(dataset_path):
        genre_path = os.path.join(dataset_path, genre_folder)
        if not os.path.isdir(genre_path):
            continue
        for file in os.listdir(genre_path):
            if not file.endswith((".mp3", ".wav")):
                continue
            out_path = os.path.join(output_dir, file.rsplit(".", 1)[0] + ".npy")
            if os.path.exists(out_path):
                continue
            tasks.append((os.path.join(genre_path, file), out_path))

    if not tasks:
        print("No new audio files to process.")
        return

    import multiprocessing
    # Use cpu_count - 2 to leave headroom for responsiveness
    num_workers = max(1, multiprocessing.cpu_count() - 2)
    print(f"Processing {len(tasks)} audio files with {num_workers} parallel workers...")
    
    with multiprocessing.Pool(num_workers) as pool:
        for _ in tqdm(pool.imap_unordered(process_single_file, tasks), total=len(tasks), desc="Extracting Audio"):
            pass


if __name__ == "__main__":
    config = load_config()
    print("Extracting audio features...")
    process_audio_dataset(config["paths"]["fma_audio"], config["paths"]["audio_features"])
    print("Audio feature extraction complete.")
