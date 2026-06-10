import librosa
import numpy as np

SR = 22050
AUDIO_LENGTH = 30
MAX_SAMPLES = SR * AUDIO_LENGTH
N_MFCC = 20
SEGMENT_LENGTH = 5
SEGMENT_SAMPLES = SR * SEGMENT_LENGTH
N_SEGMENTS = AUDIO_LENGTH // SEGMENT_LENGTH


def extract_features(audio_path):
    """Extract segment-averaged MFCC / spectral-centroid / RMS / tempo features.

    Returns a dict {mfccs, spectral_centroid, rms, tempo} matching the .npy
    format consumed by AudioFeatureDataset, or None if the file is unusable.
    """
    try:
        y, sr = librosa.load(audio_path, sr=SR)
        if len(y) > MAX_SAMPLES:
            y = y[:MAX_SAMPLES]
        else:
            y = np.pad(y, (0, MAX_SAMPLES - len(y)))

        mfccs_list, centroids_list, rms_list, tempos = [], [], [], []
        for i in range(N_SEGMENTS):
            segment = y[i * SEGMENT_SAMPLES:(i + 1) * SEGMENT_SAMPLES]
            if np.all(segment == 0):
                continue
            mfccs = librosa.feature.mfcc(y=segment, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512)
            if np.isnan(mfccs).any() or np.isinf(mfccs).any():
                continue
            mfccs_list.append(mfccs.T)
            centroids_list.append(librosa.feature.spectral_centroid(y=segment, sr=sr, hop_length=512).T)
            rms_list.append(librosa.feature.rms(y=segment, hop_length=512).T)
            tempo, _ = librosa.beat.beat_track(y=segment, sr=sr, hop_length=512)
            tempos.append(tempo)

        if not mfccs_list:
            print(f"Error: no valid segments in {audio_path}")
            return None

        mfccs = np.stack(mfccs_list, axis=0).mean(axis=0)
        spectral_centroids = np.stack(centroids_list, axis=0).mean(axis=0)
        rms = np.stack(rms_list, axis=0).mean(axis=0)
        tempo = np.mean(tempos)

        if any(np.isnan(a).any() or np.isinf(a).any() for a in (mfccs, spectral_centroids, rms)) \
                or np.isnan(tempo) or np.isinf(tempo):
            print(f"Error: features for {audio_path} contain NaN or Inf")
            return None

        mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
        spectral_centroids = spectral_centroids / 4000.0       # typical range 0-4000 Hz
        tempo = (tempo - 60) / (200 - 60)                      # typical range 60-200 BPM

        return {
            "mfccs": mfccs,                       # (time_steps, N_MFCC)
            "spectral_centroid": spectral_centroids,  # (time_steps, 1)
            "rms": rms,                           # (time_steps, 1)
            "tempo": tempo,                       # scalar
        }
    except Exception as e:
        print(f"Error processing {audio_path}: {e}")
        return None
