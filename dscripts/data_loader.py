import json
import numpy as np
import pandas as pd
from pathlib import Path

def load_audio_features():
    """Load the extracted audio features"""
    with open(Path("data") / "audio_features.json", 'r') as f:
        return json.load(f)

def load_normalized_features():
    """Load the normalized features and track IDs"""
    features = np.load(Path("data") / "normalized_features.npy")
    with open(Path("data") / "track_ids.json", 'r') as f:
        track_ids = json.load(f)
    return features, track_ids

def load_reduced_features():
    """Load the PCA-reduced features"""
    with open(Path("data") / "features_reduced.json", 'r') as f:
        return json.load(f)

def load_metadata():
    """Load the FMA metadata"""
    try:
        return pd.read_csv(Path("data") / "fma_metadata" / "echonest.csv")
    except Exception as e:
        print(f"Error loading metadata: {str(e)}")
        return None 