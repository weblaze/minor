import librosa
import numpy as np
import os
import pandas as pd
from pathlib import Path
import json
from typing import Dict, Any
import warnings
import soundfile as sf  # Add soundfile as a backup audio reader

def extract_features(audio_path: str, sr: int = 22050, duration: float = 30.0) -> Dict[str, Any]:
    """
    Extracts comprehensive audio features from an audio file.
    
    Args:
        audio_path: Path to the audio file
        sr: Sampling rate
        duration: Duration in seconds to analyze (helps standardize features)
    
    Returns:
        Dictionary containing extracted features
    """
    # Load audio with specific duration to standardize features
    try:
        # Try loading with librosa first
        try:
            y, sr = librosa.load(audio_path, sr=sr, duration=duration)
        except Exception as e:
            # If librosa fails, try soundfile as backup
            y, sr = sf.read(audio_path)
            if len(y.shape) > 1:  # Convert stereo to mono
                y = y.mean(axis=1)
            if sr != 22050:  # Resample if necessary
                y = librosa.resample(y, orig_sr=sr, target_sr=22050)
                sr = 22050
            if duration:  # Trim to duration if specified
                y = y[:int(duration * sr)]
    except Exception as e:
        raise RuntimeError(f"Error loading audio file: {str(e)}")

    # Handle empty or invalid audio files
    if len(y) == 0:
        raise ValueError("Empty audio file")

    # Calculate mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr)
    
    features = {
        # Spectral Features
        "spectral_centroid": librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        "spectral_bandwidth": librosa.feature.spectral_bandwidth(y=y, sr=sr).mean(),
        "spectral_rolloff": librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        "spectral_contrast": np.mean(librosa.feature.spectral_contrast(y=y, sr=sr), axis=1).tolist(),
        
        # Rhythm Features
        "tempo": librosa.beat.tempo(y=y, sr=sr)[0],
        "zero_crossing_rate": librosa.feature.zero_crossing_rate(y).mean(),
        
        # Mel Features
        "mfcc": np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20), axis=1).tolist(),
        "mel_spec_mean": np.mean(mel_spec, axis=1).tolist(),
        
        # Harmonic Features
        "chroma_stft": np.mean(librosa.feature.chroma_stft(y=y, sr=sr), axis=1).tolist(),
        "chroma_cens": np.mean(librosa.feature.chroma_cens(y=y, sr=sr), axis=1).tolist(),
        
        # Energy Features
        "rms": float(librosa.feature.rms(y=y).mean()),
        
        # Additional Features
        "tonnetz": np.mean(librosa.feature.tonnetz(y=y, sr=sr), axis=1).tolist(),
    }
    
    return features

def process_dataset(audio_dir: str, output_json: str, batch_size: int = 100) -> None:
    """
    Extracts features from all audio files and saves to a JSON file with batch processing.
    
    Args:
        audio_dir: Directory containing audio files
        output_json: Path to save the JSON output
        batch_size: Number of files to process before saving
    """
    audio_files = list(Path(audio_dir).rglob("*.mp3"))
    feature_data = {}
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_json), exist_ok=True)
    
    # Process files in batches
    for i, audio_file in enumerate(audio_files, 1):
        try:
            features = extract_features(str(audio_file))
            feature_data[audio_file.stem] = features
            print(f"Processed {audio_file.stem} ({i}/{len(audio_files)})")
            
            # Save batch
            if i % batch_size == 0:
                _save_batch(feature_data, output_json)
                
        except Exception as e:
            warnings.warn(f"Error processing {audio_file.stem}: {str(e)}")
            continue
    
    # Save final batch
    if feature_data:
        _save_batch(feature_data, output_json)
    
    print(f"Feature extraction complete. Saved to {output_json}")

def _save_batch(feature_data: Dict[str, Any], output_json: str) -> None:
    """Helper function to save feature data to JSON file."""
    # Load existing data if file exists
    if os.path.exists(output_json):
        with open(output_json, 'r') as f:
            existing_data = json.load(f)
        feature_data.update(existing_data)
    
    with open(output_json, 'w') as f:
        json.dump(feature_data, f, indent=4)

if __name__ == "__main__":
    AUDIO_DIR = "data/fma_small/"
    OUTPUT_JSON = "data/audio_features.json"
    process_dataset(AUDIO_DIR, OUTPUT_JSON)
