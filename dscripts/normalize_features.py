import json
import numpy as np
from sklearn.preprocessing import StandardScaler
from pathlib import Path

def normalize_features():
    # Setup paths
    data_dir = Path("data")
    input_path = data_dir / "audio_features.json"
    output_path = data_dir / "normalized_features.npy"
    track_ids_path = data_dir / "track_ids.json"  # New file to store track IDs

    # Load features from JSON
    with open(input_path, "r") as f:
        data = json.load(f)

    # Extract numerical feature values while preserving track IDs
    features = []
    track_ids = []
    
    for track_id, values in data.items():
        try:
            # Flatten the feature dictionary into a single vector
            feature_vector = []
            for value in values.values():
                if isinstance(value, list):
                    feature_vector.extend(value)
                elif isinstance(value, (int, float)):
                    feature_vector.append(value)
            features.append(feature_vector)
            track_ids.append(track_id)
        except Exception as e:
            print(f"Skipping track {track_id}: {str(e)}")

    # Convert to NumPy array
    features = np.array(features, dtype=object)
    print("Feature matrix shape before filtering:", features.shape)

    # Find the most common feature vector length
    feature_lengths = [len(f) for f in features]
    common_length = max(set(feature_lengths), key=feature_lengths.count)

    # Filter out inconsistent feature vectors
    valid_indices = [i for i, f in enumerate(features) if len(f) == common_length]
    filtered_features = np.array([features[i] for i in valid_indices], dtype=np.float32)
    filtered_track_ids = [track_ids[i] for i in valid_indices]

    print("Feature matrix shape after filtering:", filtered_features.shape)
    print(f"Retained {len(filtered_track_ids)} tracks out of {len(track_ids)}")

    # Normalize features
    scaler = StandardScaler()
    normalized_features = scaler.fit_transform(filtered_features)

    # Save normalized features and track IDs
    np.save(output_path, normalized_features)
    with open(track_ids_path, 'w') as f:
        json.dump(filtered_track_ids, f)

    print(f"✅ Features normalized and saved to {output_path}")
    print(f"✅ Track IDs saved to {track_ids_path}")
    return normalized_features, filtered_track_ids

if __name__ == "__main__":
    normalize_features() 