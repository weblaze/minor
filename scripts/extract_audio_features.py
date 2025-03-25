import os
import librosa
import numpy as np
from tqdm import tqdm

SR = 22050
AUDIO_LENGTH = 30  # Use full 30 seconds
MAX_SAMPLES = SR * AUDIO_LENGTH
N_MFCC = 20
SEGMENT_LENGTH = 5  # Process in 5-second segments
SEGMENT_SAMPLES = SR * SEGMENT_LENGTH
N_SEGMENTS = AUDIO_LENGTH // SEGMENT_LENGTH
DATASET_PATH = "datasets/fma_small"
OUTPUT_DIR = "datasets/audio_features/"
os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_temporal_features(audio_path):
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))

        # Process in segments and average
        mfccs_list = []
        for i in range(N_SEGMENTS):
            start = i * SEGMENT_SAMPLES
            end = (i + 1) * SEGMENT_SAMPLES
            segment = y[start:end]
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            mfccs_list.append(mfccs.T)
        mfccs = np.stack(mfccs_list, axis=0).mean(axis=0)  # Average over segments
        return mfccs  # Shape: (time_steps, N_MFCC)
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_dataset(dataset_path):
    for genre_folder in tqdm(os.listdir(dataset_path), desc="Processing Audio"):
        genre_path = os.path.join(dataset_path, genre_folder)
        if os.path.isdir(genre_path):
            for file in os.listdir(genre_path):
                if file.endswith((".mp3", ".wav")):
                    file_path = os.path.join(genre_path, file)
                    features = extract_temporal_features(file_path)
                    if features is not None:
                        feature_filename = file.rsplit(".", 1)[0] + ".npy"
                        np.save(os.path.join(OUTPUT_DIR, feature_filename), features)

if __name__ == "__main__":
    print("Extracting audio features...")
    process_audio_dataset(DATASET_PATH)
    print("✅ Audio feature extraction complete!")