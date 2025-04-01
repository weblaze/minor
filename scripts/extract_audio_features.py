# D:\musicc\minor\preprocessing\extract_audio_features.py
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

# Set DATASET_PATH relative to the project root (D:\musicc\minor)
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
DATASET_PATH = os.path.join(BASE_DIR, "datasets", "fma_small")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "audio_features")
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
            # Avoid all-zero segments to prevent NaN in MFCC
            if np.all(segment == 0):
                print(f"Warning: Segment {i} of {audio_path} is all zeros, skipping segment")
                continue
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            # Check for NaN or Inf in MFCCs
            if np.isnan(mfccs).any() or np.isinf(mfccs).any():
                print(f"Warning: MFCCs for segment {i} of {audio_path} contain NaN or Inf, skipping segment")
                continue
            mfccs_list.append(mfccs.T)
        
        if not mfccs_list:
            print(f"Error: No valid segments for {audio_path}")
            return None
        
        mfccs = np.stack(mfccs_list, axis=0).mean(axis=0)  # Average over segments
        # Check for NaN or Inf in averaged MFCCs
        if np.isnan(mfccs).any() or np.isinf(mfccs).any():
            print(f"Error: Averaged MFCCs for {audio_path} contain NaN or Inf")
            return None
        
        # Normalize the features to prevent large values
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        return mfccs  # Shape: (time_steps, N_MFCC)

    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None

def process_audio_dataset(dataset_path):
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Audio dataset path does not exist: {dataset_path}")
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