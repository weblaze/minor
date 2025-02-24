import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import pandas as pd
from torchvision.utils import make_grid
import torch.nn.functional as F

class TrainingAnalyzer:
    def __init__(self, output_dir="data/training_analysis"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Initialize tracking dictionaries
        self.metrics = {
            'reconstruction_loss': [],
            'style_loss': [],
            'diversity_loss': [],
            'total_loss': [],
            'epoch_avg_loss': [],
            'epoch_avg_style': [],
            'epoch_avg_diversity': []
        }
        
        # For image quality metrics
        self.image_stats = {
            'contrast': [],
            'sharpness': [],
            'color_diversity': []
        }
    
    def log_batch_metrics(self, recon_loss, style_loss, div_loss, total_loss):
        """Log metrics for each batch."""
        self.metrics['reconstruction_loss'].append(recon_loss)
        self.metrics['style_loss'].append(style_loss)
        self.metrics['diversity_loss'].append(div_loss)
        self.metrics['total_loss'].append(total_loss)
    
    def log_epoch_metrics(self, avg_loss, avg_style, avg_div):
        """Log average metrics for each epoch."""
        self.metrics['epoch_avg_loss'].append(avg_loss)
        self.metrics['epoch_avg_style'].append(avg_style)
        self.metrics['epoch_avg_diversity'].append(avg_div)
    
    def analyze_image_quality(self, generated_images):
        """Analyze quality metrics of generated images."""
        # Convert to numpy for analysis
        imgs = generated_images.detach().cpu().numpy()
        
        # Calculate contrast
        contrast = np.mean([np.std(img) for img in imgs])
        self.image_stats['contrast'].append(float(contrast))
        
        # Calculate sharpness using gradient magnitude
        # Pad the images to maintain size after gradient calculation
        padded = np.pad(imgs, ((0,0), (0,0), (0,1), (0,1)), mode='edge')
        dx = padded[:, :, 1:, :-1] - padded[:, :, :-1, :-1]
        dy = padded[:, :, :-1, 1:] - padded[:, :, :-1, :-1]
        gradients = np.sqrt(dx**2 + dy**2)
        sharpness = np.mean(gradients)
        self.image_stats['sharpness'].append(float(sharpness))
        
        # Calculate color diversity
        color_std = np.mean([np.std(img, axis=(1, 2)) for img in imgs])
        self.image_stats['color_diversity'].append(float(color_std))
    
    def plot_training_progress(self):
        """Generate plots showing training progress."""
        # Plot loss components
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.metrics['epoch_avg_loss']) + 1)
        plt.plot(epochs, self.metrics['epoch_avg_loss'], label='Total Loss')
        plt.plot(epochs, self.metrics['epoch_avg_style'], label='Style Loss')
        plt.plot(epochs, self.metrics['epoch_avg_diversity'], label='Diversity Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss Value')
        plt.title('Training Progress - Loss Components')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'training_progress.png')
        plt.close()
        
        # Plot image quality metrics
        plt.figure(figsize=(12, 6))
        epochs = range(1, len(self.image_stats['contrast']) + 1)
        plt.plot(epochs, self.image_stats['contrast'], label='Contrast')
        plt.plot(epochs, self.image_stats['sharpness'], label='Sharpness')
        plt.plot(epochs, self.image_stats['color_diversity'], label='Color Diversity')
        plt.xlabel('Epoch')
        plt.ylabel('Metric Value')
        plt.title('Image Quality Metrics Over Time')
        plt.legend()
        plt.grid(True)
        plt.savefig(self.output_dir / 'image_quality_metrics.png')
        plt.close()
    
    def save_metrics(self):
        """Save all metrics to JSON file."""
        metrics_data = {
            'training_metrics': self.metrics,
            'image_quality_metrics': self.image_stats
        }
        
        with open(self.output_dir / 'training_metrics.json', 'w') as f:
            json.dump(metrics_data, f, indent=2)
        
        # Create a summary DataFrame
        summary = pd.DataFrame({
            'Metric': ['Final Loss', 'Final Style Loss', 'Final Diversity Loss',
                      'Average Contrast', 'Average Sharpness', 'Average Color Diversity'],
            'Value': [
                self.metrics['epoch_avg_loss'][-1],
                self.metrics['epoch_avg_style'][-1],
                self.metrics['epoch_avg_diversity'][-1],
                np.mean(self.image_stats['contrast']),
                np.mean(self.image_stats['sharpness']),
                np.mean(self.image_stats['color_diversity'])
            ]
        })
        
        summary.to_csv(self.output_dir / 'training_summary.csv', index=False)
    
    def generate_report(self):
        """Generate a comprehensive analysis report."""
        report = ["# Training Analysis Report\n"]
        
        # Overall Statistics
        report.append("## Overall Statistics")
        report.append(f"- Total Epochs: {len(self.metrics['epoch_avg_loss'])}")
        report.append(f"- Final Loss: {self.metrics['epoch_avg_loss'][-1]:.4f}")
        report.append(f"- Best Loss: {min(self.metrics['epoch_avg_loss']):.4f}")
        
        # Loss Analysis
        report.append("\n## Loss Analysis")
        report.append("### Final Loss Components")
        report.append(f"- Reconstruction Loss: {self.metrics['epoch_avg_loss'][-1]:.4f}")
        report.append(f"- Style Loss: {self.metrics['epoch_avg_style'][-1]:.4f}")
        report.append(f"- Diversity Loss: {self.metrics['epoch_avg_diversity'][-1]:.4f}")
        
        # Image Quality Analysis
        report.append("\n## Image Quality Analysis")
        report.append("### Average Metrics")
        report.append(f"- Contrast: {np.mean(self.image_stats['contrast']):.4f}")
        report.append(f"- Sharpness: {np.mean(self.image_stats['sharpness']):.4f}")
        report.append(f"- Color Diversity: {np.mean(self.image_stats['color_diversity']):.4f}")
        
        # Save report
        with open(self.output_dir / 'analysis_report.md', 'w') as f:
            f.write('\n'.join(report))

def create_analyzer():
    """Create and return a new TrainingAnalyzer instance."""
    return TrainingAnalyzer() 