import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from torchvision.utils import make_grid
import torch.nn.functional as F
from sklearn.feature_selection import mutual_info_regression

class TrainingAnalyzer:
    def __init__(self, output_dir="data/training_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tracking dictionaries
        self.metrics = {
            'reconstruction_loss': [],
            'perceptual_loss': [],
            'kl_loss': [],
            'diversity_loss': [],
            'total_loss': [],
            'kl_weight': [],
            'active_dimensions': [],
            'mu_stats': {'mean': [], 'std': []},
            'logvar_stats': {'mean': [], 'std': []},
            'image_quality': {
                'contrast': [],
                'sharpness': [],
                'color_diversity': []
            }
        }
        
        # For feature importance analysis
        self.feature_importance = {
            'correlation_scores': None,
            'mutual_info_scores': None
        }
    
    def log_epoch_metrics(self, recon_loss, perceptual_loss, kl_loss, div_loss,
                         kl_weight=None, active_dims=None, mu_stats=None, logvar_stats=None):
        """Log comprehensive metrics for each epoch."""
        total_loss = recon_loss + perceptual_loss + kl_loss + div_loss
        
        self.metrics['reconstruction_loss'].append(recon_loss)
        self.metrics['perceptual_loss'].append(perceptual_loss)
        self.metrics['kl_loss'].append(kl_loss)
        self.metrics['diversity_loss'].append(div_loss)
        self.metrics['total_loss'].append(total_loss)
        
        if kl_weight is not None:
            self.metrics['kl_weight'].append(kl_weight)
        
        if active_dims is not None:
            self.metrics['active_dimensions'].append(active_dims)
        
        if mu_stats is not None:
            self.metrics['mu_stats']['mean'].append(mu_stats['mean'])
            self.metrics['mu_stats']['std'].append(mu_stats['std'])
        
        if logvar_stats is not None:
            self.metrics['logvar_stats']['mean'].append(logvar_stats['mean'])
            self.metrics['logvar_stats']['std'].append(logvar_stats['std'])
    
    def analyze_image_quality(self, generated_images):
        """Analyze quality metrics of generated images."""
        # Convert to numpy for analysis
        imgs = generated_images.detach().cpu().numpy()
        
        # Calculate contrast
        contrast = np.mean([np.std(img) for img in imgs])
        self.metrics['image_quality']['contrast'].append(float(contrast))
        
        # Calculate sharpness using gradient magnitude
        dx = np.diff(imgs, axis=2)
        dy = np.diff(imgs, axis=3)
        gradients = np.sqrt(dx**2 + dy**2)
        sharpness = np.mean(gradients)
        self.metrics['image_quality']['sharpness'].append(float(sharpness))
        
        # Calculate color diversity
        color_std = np.mean([np.std(img, axis=(1, 2)) for img in imgs])
        self.metrics['image_quality']['color_diversity'].append(float(color_std))
    
    def analyze_feature_importance(self, audio_features, generated_images):
        """Analyze the importance of different audio features for image generation."""
        # Convert to numpy arrays
        if torch.is_tensor(audio_features):
            audio_features = audio_features.detach().cpu().numpy()
        if torch.is_tensor(generated_images):
            generated_images = generated_images.detach().cpu().numpy()
        
        # Reshape images for analysis
        img_features = generated_images.reshape(generated_images.shape[0], -1)
        
        # Calculate correlation between audio features and image characteristics
        correlation_matrix = np.corrcoef(audio_features.T, img_features.T)
        n_audio_features = audio_features.shape[1]
        feature_correlations = np.abs(correlation_matrix[:n_audio_features, n_audio_features:]).mean(axis=1)
        
        # Calculate mutual information scores
        mutual_info_scores = mutual_info_regression(audio_features, img_features.mean(axis=1))
        
        # Store the results
        self.feature_importance['correlation_scores'] = feature_correlations
        self.feature_importance['mutual_info_scores'] = mutual_info_scores
        
        # Plot feature importance
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.bar(range(len(feature_correlations)), feature_correlations)
        plt.title('Feature Importance (Correlation)')
        plt.xlabel('Feature Index')
        plt.ylabel('Absolute Correlation')
        
        plt.subplot(1, 2, 2)
        plt.bar(range(len(mutual_info_scores)), mutual_info_scores)
        plt.title('Feature Importance (Mutual Information)')
        plt.xlabel('Feature Index')
        plt.ylabel('Mutual Information Score')
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'feature_importance.png')
        plt.close()
        
        # Save feature importance scores
        np.save(self.output_dir / 'feature_correlations.npy', feature_correlations)
        np.save(self.output_dir / 'mutual_info_scores.npy', mutual_info_scores)
        
        return feature_correlations, mutual_info_scores
    
    def plot_training_progress(self, latent_stats=None):
        """Generate comprehensive training progress plots."""
        # Create visualization directory
        vis_dir = self.output_dir / "visualizations"
        vis_dir.mkdir(exist_ok=True)
        
        # Plot loss components
        plt.figure(figsize=(12, 8))
        epochs = range(1, len(self.metrics['reconstruction_loss']) + 1)
        
        plt.subplot(2, 2, 1)
        plt.plot(epochs, self.metrics['reconstruction_loss'], label='Reconstruction')
        plt.plot(epochs, self.metrics['perceptual_loss'], label='Perceptual')
        plt.plot(epochs, self.metrics['kl_loss'], label='KL')
        plt.plot(epochs, self.metrics['diversity_loss'], label='Diversity')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Loss Components')
        plt.legend()
        plt.grid(True)
        
        # Plot KL weight and active dimensions
        plt.subplot(2, 2, 2)
        plt.plot(epochs, self.metrics['kl_weight'], label='KL Weight')
        if self.metrics['active_dimensions']:
            plt.plot(epochs, self.metrics['active_dimensions'], label='Active Dims %')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('KL Weight and Latent Space Activity')
        plt.legend()
        plt.grid(True)
        
        # Plot latent space statistics
        plt.subplot(2, 2, 3)
        plt.plot(epochs, self.metrics['mu_stats']['mean'], label='μ mean')
        plt.plot(epochs, self.metrics['mu_stats']['std'], label='μ std')
        plt.plot(epochs, self.metrics['logvar_stats']['mean'], label='logvar mean')
        plt.plot(epochs, self.metrics['logvar_stats']['std'], label='logvar std')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Latent Space Statistics')
        plt.legend()
        plt.grid(True)
        
        # Plot image quality metrics
        plt.subplot(2, 2, 4)
        plt.plot(epochs, self.metrics['image_quality']['contrast'], label='Contrast')
        plt.plot(epochs, self.metrics['image_quality']['sharpness'], label='Sharpness')
        plt.plot(epochs, self.metrics['image_quality']['color_diversity'], label='Color Div')
        plt.xlabel('Epoch')
        plt.ylabel('Value')
        plt.title('Image Quality Metrics')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.savefig(vis_dir / 'training_progress.png')
        plt.close()
        
        # Plot latent dimension activity heatmap if available
        if latent_stats and 'z_activity' in latent_stats:
            plt.figure(figsize=(12, 4))
            activity_matrix = np.array(latent_stats['z_activity'])
            sns.heatmap(activity_matrix.T, cmap='YlOrRd')
            plt.xlabel('Epoch')
            plt.ylabel('Latent Dimension')
            plt.title('Latent Dimension Activity Over Time')
            plt.savefig(vis_dir / 'latent_activity.png')
            plt.close()
    
    def save_metrics(self, latent_stats=None):
        """Save all metrics to JSON file with enhanced statistics."""
        metrics_data = {
            'training_metrics': self.metrics,
            'final_stats': {
                'reconstruction_loss': np.mean(self.metrics['reconstruction_loss'][-5:]),
                'perceptual_loss': np.mean(self.metrics['perceptual_loss'][-5:]),
                'kl_loss': np.mean(self.metrics['kl_loss'][-5:]),
                'diversity_loss': np.mean(self.metrics['diversity_loss'][-5:]),
                'active_dimensions': np.mean(self.metrics['active_dimensions'][-5:]) if self.metrics['active_dimensions'] else None,
                'image_quality': {
                    'contrast': np.mean(self.metrics['image_quality']['contrast'][-5:]),
                    'sharpness': np.mean(self.metrics['image_quality']['sharpness'][-5:]),
                    'color_diversity': np.mean(self.metrics['image_quality']['color_diversity'][-5:])
                }
            }
        }
        
        if latent_stats:
            metrics_data['latent_space_analysis'] = {
                'final_active_dims': int(np.sum(latent_stats['z_activity'][-1] > 0.1)),
                'mean_active_dims': float(np.mean([np.sum(act > 0.1) for act in latent_stats['z_activity']])),
                'mu_stats': {
                    'mean': float(np.mean(latent_stats['mu_mean'][-5:])),
                    'std': float(np.mean(latent_stats['mu_std'][-5:]))
                },
                'logvar_stats': {
                    'mean': float(np.mean(latent_stats['logvar_mean'][-5:])),
                    'std': float(np.mean(latent_stats['logvar_std'][-5:]))
                }
            }
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
    
    def generate_report(self, latent_stats=None):
        """Generate a comprehensive analysis report with latent space insights."""
        report = ["# Training Analysis Report\n"]
        
        # Overall Statistics
        report.append("## Overall Statistics")
        report.append(f"- Total Epochs: {len(self.metrics['reconstruction_loss'])}")
        report.append(f"- Final Reconstruction Loss: {self.metrics['reconstruction_loss'][-1]:.4f}")
        report.append(f"- Final KL Loss: {self.metrics['kl_loss'][-1]:.4f}")
        report.append(f"- Final Perceptual Loss: {self.metrics['perceptual_loss'][-1]:.4f}")
        report.append(f"- Final Diversity Loss: {self.metrics['diversity_loss'][-1]:.4f}")
        
        # Latent Space Analysis
        if latent_stats:
            report.append("\n## Latent Space Analysis")
            report.append("### Final State")
            report.append(f"- Active Dimensions: {int(np.sum(latent_stats['z_activity'][-1] > 0.1))}")
            report.append(f"- Mean μ: {latent_stats['mu_mean'][-1]:.4f} ± {latent_stats['mu_std'][-1]:.4f}")
            report.append(f"- Mean logvar: {latent_stats['logvar_mean'][-1]:.4f} ± {latent_stats['logvar_std'][-1]:.4f}")
            
            report.append("\n### Training Dynamics")
            report.append("- Latent Space Evolution:")
            report.append(f"  - Initial active dims: {int(np.sum(latent_stats['z_activity'][0] > 0.1))}")
            report.append(f"  - Peak active dims: {int(max(np.sum(act > 0.1) for act in latent_stats['z_activity']))}")
            report.append(f"  - Final active dims: {int(np.sum(latent_stats['z_activity'][-1] > 0.1))}")
        
        # Image Quality Analysis
        report.append("\n## Image Quality Analysis")
        report.append("### Final Metrics")
        report.append(f"- Contrast: {self.metrics['image_quality']['contrast'][-1]:.4f}")
        report.append(f"- Sharpness: {self.metrics['image_quality']['sharpness'][-1]:.4f}")
        report.append(f"- Color Diversity: {self.metrics['image_quality']['color_diversity'][-1]:.4f}")
        
        # Training Recommendations
        report.append("\n## Training Recommendations")
        
        # Analyze KL loss trend
        kl_trend = np.array(self.metrics['kl_loss'])
        if np.mean(kl_trend[-5:]) < 0.001:
            report.append("- Consider increasing KL weight to prevent posterior collapse")
        elif np.mean(kl_trend[-5:]) > 0.1:
            report.append("- Consider decreasing KL weight to improve reconstruction")
        
        # Analyze active dimensions
        if latent_stats and np.mean(latent_stats['z_activity'][-1]) < 0.3:
            report.append("- Consider techniques to increase latent space utilization")
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))

def create_analyzer():
    """Create and return a new TrainingAnalyzer instance."""
    return TrainingAnalyzer() 