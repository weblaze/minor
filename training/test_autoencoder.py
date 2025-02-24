import torch
import numpy as np
from PIL import Image
from pathlib import Path
from models.autoencoder.model import AudioToImageAutoencoder
from torchvision.utils import save_image
import random

def load_model(model_path, device):
    model = AudioToImageAutoencoder().to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model

def generate_images(model, audio_features, output_dir, num_samples=5):
    """Generate images from audio features using the trained model."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    # Randomly select audio features if we want fewer than all
    if num_samples < len(audio_features):
        indices = random.sample(range(len(audio_features)), num_samples)
        selected_features = audio_features[indices]
    else:
        selected_features = audio_features
    
    # Convert to tensor
    features_tensor = torch.FloatTensor(selected_features).to(next(model.parameters()).device)
    
    # Generate images
    with torch.no_grad():
        generated_images, _, _ = model(features_tensor)  # Unpack the tuple
    
    # Save images with enhanced contrast
    for i, image in enumerate(generated_images):
        # Apply additional post-processing
        enhanced = image.cpu()
        # Adjust contrast
        enhanced = (enhanced - enhanced.mean()) * 1.2 + 0.5
        enhanced = enhanced.clamp(0, 1)
        
        save_image(enhanced, output_dir / f"generated_image_{i+1}.png")
        print(f"Saved image {i+1} to {output_dir}/generated_image_{i+1}.png")

def main():
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
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
    
    print("Done! Check the generated images in the output directory.")

if __name__ == "__main__":
    main() 