import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Any
from dscripts.logger import DataLogger, get_logger

class FeatureAnalyzer:
    def __init__(self, log_dir: Path = None):
        self.log_dir = Path("logs") if log_dir is None else Path(log_dir)
        self.logger = get_logger("feature_analyzer", self.log_dir)
        self.data_logger = DataLogger("feature_analysis", self.log_dir)
        
        # Create visualization directory
        self.vis_dir = self.log_dir / "visualizations"
        self.vis_dir.mkdir(exist_ok=True)
    
    def analyze_features(self, features_path: Path, reduced_features_path: Path):
        """Analyze both raw and reduced features."""
        self.logger.info("Starting feature analysis")
        
        # Load features
        raw_features = self._load_json(features_path)
        reduced_features = self._load_json(reduced_features_path)
        
        # Analyze raw features
        raw_stats = self._analyze_raw_features(raw_features)
        self.data_logger.metrics["raw_features"] = raw_stats
        
        # Analyze reduced features
        reduced_stats = self._analyze_reduced_features(reduced_features)
        self.data_logger.metrics["reduced_features"] = reduced_stats
        
        # Generate visualizations
        self._generate_visualizations(raw_features, reduced_features)
        
        # Save analysis results
        self.data_logger.save_metrics()
        self.logger.info("Feature analysis completed")
    
    def _load_json(self, path: Path) -> Dict[str, Any]:
        """Load JSON file with error handling."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading {path}: {str(e)}")
            raise
    
    def _analyze_raw_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze raw audio features."""
        stats = {
            "num_tracks": len(features),
            "feature_types": list(next(iter(features.values())).keys()),
            "statistics": {}
        }
        
        # Calculate statistics for each feature type
        for feature_type in stats["feature_types"]:
            values = []
            for track_features in features.values():
                if isinstance(track_features[feature_type], (int, float)):
                    values.append(track_features[feature_type])
                elif isinstance(track_features[feature_type], list):
                    values.extend(track_features[feature_type])
            
            if values:
                stats["statistics"][feature_type] = {
                    "mean": float(np.mean(values)),
                    "std": float(np.std(values)),
                    "min": float(np.min(values)),
                    "max": float(np.max(values))
                }
        
        return stats
    
    def _analyze_reduced_features(self, features: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze reduced features."""
        # Convert to numpy array for analysis
        feature_matrix = np.array([features[track_id] for track_id in features])
        
        stats = {
            "num_tracks": len(features),
            "num_dimensions": feature_matrix.shape[1],
            "mean": float(np.mean(feature_matrix)),
            "std": float(np.std(feature_matrix)),
            "sparsity": float(np.mean(feature_matrix == 0)),
            "dimension_stats": {
                i: {
                    "mean": float(np.mean(feature_matrix[:, i])),
                    "std": float(np.std(feature_matrix[:, i]))
                }
                for i in range(feature_matrix.shape[1])
            }
        }
        
        return stats
    
    def _generate_visualizations(self, raw_features: Dict[str, Any], 
                               reduced_features: Dict[str, Any]):
        """Generate visualizations for feature analysis."""
        # Plot distribution of raw scalar features
        scalar_features = {}
        for track_id, features in raw_features.items():
            for name, value in features.items():
                if isinstance(value, (int, float)):
                    if name not in scalar_features:
                        scalar_features[name] = []
                    scalar_features[name].append(value)
        
        plt.figure(figsize=(15, 5))
        plt.title("Distribution of Raw Scalar Features")
        plt.boxplot(list(scalar_features.values()), labels=list(scalar_features.keys()))
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(self.vis_dir / "raw_feature_distributions.png")
        plt.close()
        
        # Plot reduced feature distributions
        reduced_matrix = np.array([reduced_features[track_id] 
                                 for track_id in reduced_features])
        
        plt.figure(figsize=(15, 5))
        plt.title("Distribution of Reduced Features")
        plt.boxplot([reduced_matrix[:, i] for i in range(min(10, reduced_matrix.shape[1]))])
        plt.xlabel("Feature Component")
        plt.ylabel("Value")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "reduced_feature_distributions.png")
        plt.close()
        
        # Plot correlation matrix of reduced features
        plt.figure(figsize=(10, 10))
        corr_matrix = np.corrcoef(reduced_matrix.T)
        sns.heatmap(corr_matrix[:10, :10], annot=True, cmap='coolwarm')
        plt.title("Correlation Matrix of First 10 Reduced Features")
        plt.tight_layout()
        plt.savefig(self.vis_dir / "reduced_feature_correlations.png")
        plt.close()

def main():
    # Setup paths
    data_dir = Path("data")
    features_path = data_dir / "audio_features.json"
    reduced_features_path = data_dir / "features_reduced.json"
    
    # Create analyzer and process
    analyzer = FeatureAnalyzer()
    analyzer.analyze_features(features_path, reduced_features_path)

if __name__ == "__main__":
    main() 