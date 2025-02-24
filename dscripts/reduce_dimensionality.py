import json
import numpy as np
import pickle
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.pipeline import Pipeline
from pathlib import Path
import matplotlib.pyplot as plt
from typing import Tuple, Dict, Any
from dscripts.logger import DataLogger, get_logger

class FeatureReducer:
    def __init__(self, n_components: int = 128, log_dir: Path = None):
        self.n_components = n_components
        self.log_dir = Path("logs") if log_dir is None else Path(log_dir)
        self.logger = get_logger("feature_reducer", self.log_dir)
        self.data_logger = DataLogger("feature_reduction", self.log_dir)
        
        # Initialize models
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=self.n_components)
        self.tsne = TSNE(n_components=2, random_state=42)
        
        # For storing results
        self.reduced_features = None
        self.feature_importance = None
    
    def reduce_dimensionality(self, input_path: Path, track_ids_path: Path, 
                            output_path: Path, model_path: Path) -> None:
        """
        Reduce dimensionality of features while preserving important information.
        """
        self.logger.info("Starting dimensionality reduction")
        
        # Load data
        features, track_ids = self._load_data(input_path, track_ids_path)
        self.logger.info(f"Loaded features shape: {features.shape}")
        
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        self.logger.info("Features scaled")
        
        # Apply PCA
        self.reduced_features = self.pca.fit_transform(features_scaled)
        self.logger.info(f"Reduced features shape: {self.reduced_features.shape}")
        
        # Calculate and log explained variance
        explained_variance = np.cumsum(self.pca.explained_variance_ratio_)
        self.data_logger.log_metric("explained_variance", float(explained_variance[-1]))
        self.data_logger.log_metric("n_components_95", 
                                  int(np.argmax(explained_variance >= 0.95) + 1))
        
        # Calculate feature importance
        self.feature_importance = self._calculate_feature_importance()
        
        # Generate visualizations
        self._generate_visualizations()
        
        # Save results
        self._save_results(output_path, model_path, track_ids)
        
        self.logger.info("Dimensionality reduction completed")
    
    def _load_data(self, input_path: Path, track_ids_path: Path) -> Tuple[np.ndarray, list]:
        """Load and validate input data."""
        try:
            features = np.load(input_path)
            with open(track_ids_path, 'r') as f:
                track_ids = json.load(f)
            
            if len(features) != len(track_ids):
                raise ValueError("Mismatch between features and track IDs")
            
            return features, track_ids
            
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise
    
    def _calculate_feature_importance(self) -> Dict[str, float]:
        """Calculate importance of each feature component."""
        importance = np.abs(self.pca.components_).mean(axis=0)
        return {f"component_{i}": float(imp) for i, imp in enumerate(importance)}
    
    def _generate_visualizations(self):
        """Generate visualizations of the reduced features."""
        # Create visualization directory
        vis_dir = self.log_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Plot explained variance
        plt.figure(figsize=(10, 5))
        plt.plot(np.cumsum(self.pca.explained_variance_ratio_))
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.title('Explained Variance vs. Number of Components')
        plt.grid(True)
        plt.savefig(vis_dir / 'explained_variance.png')
        plt.close()
        
        # Generate t-SNE visualization
        tsne_result = self.tsne.fit_transform(self.reduced_features)
        
        plt.figure(figsize=(10, 10))
        plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5)
        plt.title('t-SNE Visualization of Reduced Features')
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.savefig(vis_dir / 'tsne_visualization.png')
        plt.close()
    
    def _save_results(self, output_path: Path, model_path: Path, track_ids: list):
        """Save reduced features and model."""
        try:
            # Save reduced features
            reduced_features_dict = {
                track_id: [float(val) for val in vec]
                for track_id, vec in zip(track_ids, self.reduced_features)
            }
            
            with open(output_path, "w") as f:
                json.dump(reduced_features_dict, f, indent=2)
            
            # Save models
            with open(model_path, "wb") as f:
                pickle.dump({
                    'scaler': self.scaler,
                    'pca': self.pca,
                    'feature_importance': self.feature_importance
                }, f)
            
            # Log feature statistics
            self.data_logger.log_stats(self.reduced_features, "reduced_features")
            self.data_logger.metrics["feature_importance"] = self.feature_importance
            self.data_logger.save_metrics()
            
            self.logger.info(f"Saved reduced features to {output_path}")
            self.logger.info(f"Saved models to {model_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving results: {str(e)}")
            raise

def main():
    # Setup paths
    data_dir = Path("data")
    input_path = data_dir / "normalized_features.npy"
    track_ids_path = data_dir / "track_ids.json"
    output_path = data_dir / "features_reduced.json"
    model_path = data_dir / "pca_model.pkl"

    # Create reducer and process
    reducer = FeatureReducer(n_components=128)
    reducer.reduce_dimensionality(input_path, track_ids_path, output_path, model_path)

if __name__ == "__main__":
    main() 