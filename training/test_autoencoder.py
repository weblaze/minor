import torch
import numpy as np
from PIL import Image
from pathlib import Path
from models.autoencoder.model import AudioToImageAutoencoder
from torchvision.utils import save_image
import random

# Set CUDA configurations for better GPU performance
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.deterministic = False

def load_model(model_path, device):
    """Load model from checkpoint."""
    # Initialize model and move to device
    model = AudioToImageAutoencoder().to(device)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Handle different state dict formats
    if 'model_state_dict' in checkpoint:
        # Loading from training checkpoint
        state_dict = checkpoint['model_state_dict']
    else:
        # Loading from model-only save
        state_dict = checkpoint
    
    # Handle DataParallel wrapped state dict
    if list(state_dict.keys())[0].startswith('module.'):
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'loss' in checkpoint:
        print(f"Model loss at checkpoint: {checkpoint['loss']:.4f}")
    
    return model

def generate_images(model, audio_features, output_dir, num_samples=5):
    """Generate images from audio features using the trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Get the device of the model
    device = next(model.parameters()).device
    
    # Randomly select audio features if we want fewer than all
    if num_samples < len(audio_features):
        indices = random.sample(range(len(audio_features)), num_samples)
        selected_features = audio_features[indices]
    else:
        selected_features = audio_features
    
    # Convert to tensor and move to device
    features_tensor = torch.FloatTensor(selected_features).to(device, non_blocking=True)
    
    # Print feature statistics
    print(f"Input feature stats - Min: {features_tensor.min():.4f}, Max: {features_tensor.max():.4f}, Mean: {features_tensor.mean():.4f}")
    
    # Generate images with CUDA optimization
    with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
        generated_images, mu, logvar = model(features_tensor)
        print(f"Raw output stats - Min: {generated_images.min():.4f}, Max: {generated_images.max():.4f}, Mean: {generated_images.mean():.4f}")
    
    # Save images with enhanced visualization
    for i, image in enumerate(generated_images):
        # Move to CPU and detach for processing
        image = image.cpu().detach()
        
        print(f"\nImage {i+1} stats:")
        print(f"Before enhancement - Min: {image.min():.4f}, Max: {image.max():.4f}, Mean: {image.mean():.4f}")
        
        # Convert to numpy for histogram equalization
        image_np = image.permute(1, 2, 0).numpy()
        
        # 1. Basic normalization to full range
        enhanced1 = torch.tensor((image_np - image_np.min()) / (image_np.max() - image_np.min())).permute(2, 0, 1)
        
        # 2. Adaptive histogram-like enhancement
        # First normalize to 0-1
        normalized = (image - image.min()) / (image.max() - image.min())
        # Then apply gamma correction to boost mid-tones
        enhanced2 = torch.pow(normalized, 0.4)  # gamma = 0.4 for stronger boost
        
        # 3. Extreme contrast enhancement for early training visualization
        mean = image.mean()
        std = image.std()
        enhanced3 = (image - mean) / (std + 1e-5)  # Normalize by standard deviation
        enhanced3 = torch.tanh(enhanced3 * 2) * 0.5 + 0.5  # Map to 0-1 range using tanh with stronger scaling
        
        # Save all versions
        save_image(enhanced1, output_dir / f"generated_image_{i+1}_norm.png")
        save_image(enhanced2, output_dir / f"generated_image_{i+1}_gamma.png")
        save_image(enhanced3, output_dir / f"generated_image_{i+1}_contrast.png")
        
        print(f"After enhancement - Saved three versions of image {i+1}")
        print(f"Normalized version stats - Min: {enhanced1.min():.4f}, Max: {enhanced1.max():.4f}, Mean: {enhanced1.mean():.4f}")
        print(f"Gamma version stats - Min: {enhanced2.min():.4f}, Max: {enhanced2.max():.4f}, Mean: {enhanced2.mean():.4f}")
        print(f"Contrast version stats - Min: {enhanced3.min():.4f}, Max: {enhanced3.max():.4f}, Mean: {enhanced3.mean():.4f}")
        
        # Also save the raw output for reference
        save_image(image, output_dir / f"generated_image_{i+1}_raw.png")

def main():
    # Set device with proper CUDA memory management
    if torch.cuda.is_available():
        device = torch.device("cuda")
        torch.cuda.empty_cache()  # Clear GPU cache
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
    else:
        device = torch.device("cpu")
        print("CUDA is not available. Using CPU")
    
    # Paths
    model_path = "models/autoencoder/checkpoint_epoch_4.pth"  # Use latest checkpoint
    audio_features_path = "data/normalized_features.npy"  # Use normalized features
    output_dir = "data/generated_images/epoch_4"  # Separate directory for this checkpoint
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load audio features
    print("Loading audio features...")
    audio_features = np.load(audio_features_path)
    
    # Generate images
    print("Generating images...")
    generate_images(model, audio_features, output_dir, num_samples=10)  # Generate 10 samples
    
    # Clean up GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("Done! Check the generated images in the output directory.")

if __name__ == "__main__":
    main() 