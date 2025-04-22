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
        spectral_centroids_list = []
        rms_list = []
        tempos = []
        for i in range(N_SEGMENTS):
            start = i * SEGMENT_SAMPLES
            end = (i + 1) * SEGMENT_SAMPLES
            segment = y[start:end]
            # Avoid all-zero segments to prevent NaN in MFCC
            if np.all(segment == 0):
                print(f"Warning: Segment {i} of {audio_path} is all zeros, skipping segment")
                continue
            # Extract MFCCs
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            if np.isnan(mfccs).any() or np.isinf(mfccs).any():
                print(f"Warning: MFCCs for segment {i} of {audio_path} contain NaN or Inf, skipping segment")
                continue
            mfccs_list.append(mfccs.T)
            # Extract spectral centroid
            spectral_centroid = librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=512)
            spectral_centroids_list.append(spectral_centroid.T)
            # Extract RMS
            rms = librosa.feature.rms(y=segment, hop_length=512)
            rms_list.append(rms.T)
            # Extract tempo
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr, hop_length=512)
            tempos.append(tempo)

        if not mfccs_list:
            print(f"Error: No valid segments for {audio_path}")
            return None

        # Average over segments
        mfccs = np.stack(mfccs_list, axis=0).mean(axis=0)  # Shape: (time_steps, N_MFCC)
        spectral_centroids = np.stack(spectral_centroids_list, axis=0).mean(axis=0)  # Shape: (time_steps, 1)
        rms = np.stack(rms_list, axis=0).mean(axis=0)  # Shape: (time_steps, 1)
        tempo = np.mean(tempos)  # Scalar

        # Check for NaN or Inf in features
        if (np.isnan(mfccs).any() or np.isinf(mfccs).any() or
            np.isnan(spectral_centroids).any() or np.isinf(spectral_centroids).any() or
            np.isnan(rms).any() or np.isinf(rms).any() or
            np.isnan(tempo) or np.isinf(tempo)):
            print(f"Error: Features for {audio_path} contain NaN or Inf")
            return None

        # Normalize the features
        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        spectral_centroids = spectral_centroids / 4000.0  # Normalize to [0, 1] (typical range 0-4000 Hz)
        rms = rms  # Already in [0, 1]
        tempo = (tempo - 60) / (200 - 60)  # Normalize to [0, 1] (typical range 60-200 BPM)

        # Combine features into a dictionary
        features = {
            'mfccs': mfccs,  # Shape: (time_steps, N_MFCC)
            'spectral_centroid': spectral_centroids,  # Shape: (time_steps, 1)
            'rms': rms,  # Shape: (time_steps, 1)
            'tempo': tempo  # Scalar
        }
        return features

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