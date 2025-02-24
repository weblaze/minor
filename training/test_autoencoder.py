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
    # Initialize model and move to device
    model = AudioToImageAutoencoder().to(device)
    
    # Load state dict with proper device mapping
    state_dict = torch.load(model_path, map_location=device)
    
    # Handle DataParallel wrapped state dict
    if list(state_dict.keys())[0].startswith('module.'):
        # Create new OrderedDict without 'module.' prefix
        from collections import OrderedDict
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove 'module.' prefix
            new_state_dict[name] = v
        state_dict = new_state_dict
    
    model.load_state_dict(state_dict)
    model.eval()
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
    
    # Generate images with CUDA optimization
    with torch.cuda.amp.autocast() if device.type == 'cuda' else torch.no_grad():
        generated_images, _, _ = model(features_tensor)
    
    # Save images with enhanced contrast
    for i, image in enumerate(generated_images):
        # Move to CPU for saving
        enhanced = image.cpu()
        # Adjust contrast
        enhanced = (enhanced - enhanced.mean()) * 1.2 + 0.5
        enhanced = enhanced.clamp(0, 1)
        
        save_image(enhanced, output_dir / f"generated_image_{i+1}.png")
        print(f"Saved image {i+1} to {output_dir}/generated_image_{i+1}.png")

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
    model_path = "models/autoencoder/audio_to_image_autoencoder.pth"
    audio_features_path = "data/normalized_features.npy"
    output_dir = "data/generated_images"
    
    # Load model
    print("Loading model...")
    model = load_model(model_path, device)
    
    # Load audio features
    print("Loading audio features...")
    audio_features = np.load(audio_features_path)
    
    # Generate images
    print("Generating images...")
    generate_images(model, audio_features, output_dir, num_samples=5)
    
    # Clean up GPU memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    print("Done! Check the generated images in the output directory.")

if __name__ == "__main__":
    main() 