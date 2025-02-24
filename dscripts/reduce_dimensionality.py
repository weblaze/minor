import json
import numpy as np
import pickle
from sklearn.decomposition import PCA
from pathlib import Path

def reduce_dimensionality():
    # Setup paths
    data_dir = Path("data")
    input_path = data_dir / "normalized_features.npy"
    track_ids_path = data_dir / "track_ids.json"
    output_path = data_dir / "features_reduced.json"
    model_path = data_dir / "pca_model.pkl"

    # Load normalized features and track IDs
    features = np.load(input_path)
    with open(track_ids_path, 'r') as f:
        track_ids = json.load(f)

    print(f"Loaded features shape: {features.shape}")
    print(f"Number of tracks: {len(track_ids)}")

    # Apply PCA to reduce to 64 dimensions
    n_components = min(64, features.shape[1])
    pca = PCA(n_components=n_components)
    features_reduced = pca.fit_transform(features)

    # Calculate explained variance
    explained_variance = np.sum(pca.explained_variance_ratio_)
    print(f"Explained variance ratio: {explained_variance:.3f}")

    # Save the reduced features in JSON format
    # Convert float32 to regular Python float for JSON serialization
    reduced_features_dict = {
        track_id: [float(val) for val in vec]
        for track_id, vec in zip(track_ids, features_reduced)
    }
    
    with open(output_path, "w") as f:
        json.dump(reduced_features_dict, f)

    # Save the PCA model for inference
    with open(model_path, "wb") as f:
        pickle.dump(pca, f)

    print(f"✅ Features reduced to {n_components} dimensions")
    print(f"✅ Saved reduced features to {output_path}")
    print(f"✅ Saved PCA model to {model_path}")

if __name__ == "__main__":
    reduce_dimensionality() 