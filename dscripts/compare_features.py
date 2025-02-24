import numpy as np
import json
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import pairwise_distances
from scipy.stats import pearsonr
import pandas as pd
from sklearn.decomposition import PCA

def load_all_features():
    """Load features from all three stages of processing."""
    data_dir = Path("data")
    
    # Load original extracted features
    with open(data_dir / "audio_features.json", 'r') as f:
        raw_features = json.load(f)
    
    # Load normalized features and track IDs
    normalized_features = np.load(data_dir / "normalized_features.npy")
    with open(data_dir / "track_ids.json", 'r') as f:
        track_ids = json.load(f)
    
    # Load PCA reduced features and model
    with open(data_dir / "features_reduced.json", 'r') as f:
        pca_features = json.load(f)
    with open(data_dir / "pca_model.pkl", 'rb') as f:
        import pickle
        pca_model = pickle.load(f)
    
    return raw_features, normalized_features, pca_features, track_ids, pca_model

def prepare_feature_matrices(raw_features, track_ids):
    """Convert raw features dictionary to matrix format."""
    feature_vectors = []
    valid_track_ids = []
    
    for track_id in track_ids:
        if track_id in raw_features:
            # Flatten the feature dictionary into a vector
            feature_vector = []
            for value in raw_features[track_id].values():
                if isinstance(value, list):
                    feature_vector.extend(value)
                else:
                    feature_vector.append(value)
            feature_vectors.append(feature_vector)
            valid_track_ids.append(track_id)
    
    return np.array(feature_vectors), valid_track_ids

def plot_feature_distributions(raw_matrix, normalized_features, pca_features, save_path):
    """Plot distribution of features at each processing stage."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Raw features distribution
    sns.boxplot(data=raw_matrix, ax=axes[0], orient='h', whis=1.5)
    axes[0].set_title('Raw Features')
    axes[0].set_xlabel('Value')
    
    # Normalized features distribution
    sns.boxplot(data=normalized_features, ax=axes[1], orient='h', whis=1.5)
    axes[1].set_title('Normalized Features')
    axes[1].set_xlabel('Value')
    
    # PCA features distribution
    pca_matrix = np.array(list(pca_features.values()))
    sns.boxplot(data=pca_matrix, ax=axes[2], orient='h', whis=1.5)
    axes[2].set_title('PCA Reduced Features')
    axes[2].set_xlabel('Value')
    
    plt.tight_layout()
    plt.savefig(save_path / 'feature_distributions.png')
    plt.close()

def compute_pairwise_correlations(features1, features2):
    """Compute correlation between corresponding dimensions."""
    correlations = []
    for i in range(min(features1.shape[1], features2.shape[1])):
        if i < features2.shape[1]:
            corr, _ = pearsonr(features1[:, i], features2[:, i])
            correlations.append(corr)
    return np.mean(correlations), np.std(correlations)

def analyze_dimensionality_impact(raw_matrix, normalized_features, pca_matrix):
    """Analyze the impact of dimensionality reduction."""
    results = {
        'Stage': ['Raw', 'Normalized', 'PCA'],
        'Dimensions': [raw_matrix.shape[1], normalized_features.shape[1], pca_matrix.shape[1]],
        'Mean': [np.mean(raw_matrix), np.mean(normalized_features), np.mean(pca_matrix)],
        'Std': [np.std(raw_matrix), np.std(normalized_features), np.std(pca_matrix)],
        'Sparsity': [
            np.mean(raw_matrix == 0),
            np.mean(normalized_features == 0),
            np.mean(pca_matrix == 0)
        ]
    }
    return pd.DataFrame(results)

def plot_correlation_heatmap(raw_matrix, normalized_features, pca_matrix, save_path):
    """Plot correlation heatmap between different processing stages."""
    # Sample a subset of features for visualization
    n_samples = min(50, raw_matrix.shape[1], normalized_features.shape[1], pca_matrix.shape[1])
    
    correlations = np.zeros((3, 3))
    labels = ['Raw', 'Normalized', 'PCA']
    
    features = [
        raw_matrix[:, :n_samples],
        normalized_features[:, :n_samples],
        pca_matrix[:, :n_samples]
    ]
    
    for i in range(3):
        for j in range(3):
            if i == j:
                correlations[i, j] = 1.0
            else:
                corr_mean, _ = compute_pairwise_correlations(features[i], features[j])
                correlations[i, j] = corr_mean
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(correlations, annot=True, cmap='coolwarm', xticklabels=labels, yticklabels=labels)
    plt.title('Correlation between Processing Stages')
    plt.tight_layout()
    plt.savefig(save_path / 'correlation_heatmap.png')
    plt.close()

def analyze_pca_components(normalized_features, pca_model, save_path):
    """Analyze PCA components and their importance."""
    # Plot explained variance ratio
    plt.figure(figsize=(10, 5))
    cumsum = np.cumsum(pca_model.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum) + 1), cumsum, 'b-')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.title('Explained Variance vs. Number of Components')
    plt.grid(True)
    plt.savefig(save_path / 'pca_explained_variance.png')
    plt.close()
    
    # Get feature importance
    component_importance = np.abs(pca_model.components_).mean(axis=0)
    top_features = np.argsort(component_importance)[-10:][::-1]
    
    return {
        'total_variance_explained': np.sum(pca_model.explained_variance_ratio_),
        'n_components_95': np.argmax(cumsum >= 0.95) + 1,
        'top_feature_importance': component_importance[top_features]
    }

def main():
    # Create output directory for visualizations
    output_dir = Path("data/analysis")
    output_dir.mkdir(exist_ok=True)
    
    # Load all features
    print("Loading features...")
    raw_features, normalized_features, pca_features, track_ids, pca_model = load_all_features()
    
    # Prepare feature matrices
    print("Preparing feature matrices...")
    raw_matrix, valid_track_ids = prepare_feature_matrices(raw_features, track_ids)
    pca_matrix = np.array([pca_features[tid] for tid in valid_track_ids])
    
    # Generate visualizations
    print("Generating visualizations...")
    plot_feature_distributions(raw_matrix, normalized_features, pca_features, output_dir)
    plot_correlation_heatmap(raw_matrix, normalized_features, pca_matrix, output_dir)
    
    # Analyze PCA components
    print("\nAnalyzing PCA components...")
    pca_analysis = analyze_pca_components(normalized_features, pca_model, output_dir)
    print(f"Total variance explained by PCA: {pca_analysis['total_variance_explained']:.3f}")
    print(f"Components needed for 95% variance: {pca_analysis['n_components_95']}")
    print("\nTop feature importances:")
    print(pca_analysis['top_feature_importance'])
    
    # Analyze dimensionality impact
    print("\nAnalyzing dimensionality impact...")
    analysis_df = analyze_dimensionality_impact(raw_matrix, normalized_features, pca_matrix)
    print("\nFeature Analysis Summary:")
    print(analysis_df.to_string(index=False))
    
    # Save analysis results
    analysis_df.to_csv(output_dir / 'feature_analysis.csv', index=False)
    
    # Compute and display correlations between stages
    print("\nCorrelation Analysis:")
    stages = ['Raw-Normalized', 'Normalized-PCA', 'Raw-PCA']
    feature_sets = [(raw_matrix, normalized_features), 
                   (normalized_features, pca_matrix),
                   (raw_matrix, pca_matrix)]
    
    for stage, (features1, features2) in zip(stages, feature_sets):
        mean_corr, std_corr = compute_pairwise_correlations(features1, features2)
        print(f"{stage} correlation: {mean_corr:.3f} ± {std_corr:.3f}")

if __name__ == "__main__":
    main() 