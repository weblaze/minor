import os
import sys
import yaml
import torch
from torch.utils.data import DataLoader
import torchvision.utils as vutils
from PIL import Image

# Setup base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.datasets import ImageDataset

def test_reconstruction():
    # 1. Load configuration
    config_path = os.path.join(BASE_DIR, "configs", "config.yaml")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    sys_config = config['system']
    img_config = config['image_vae']
    
    device = torch.device(sys_config['device'] if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Paths
    IMAGE_PATH = os.path.join(BASE_DIR, config['paths']['image_features'])
    IMAGE_MODEL_PATH = os.path.join(BASE_DIR, config['paths']['models_dir'], "image_vae.pth")
    OUTPUT_DIR = os.path.join(BASE_DIR, "evaluation", "results")
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 3. Initialize Model
    latent_channels = config['latent_diffusion'].get('latent_channels', 8)
    model = ImageVAE(latent_channels=latent_channels).to(device)
    
    if not os.path.exists(IMAGE_MODEL_PATH):
        print(f"Warning: Checkpoint not found at {IMAGE_MODEL_PATH}. You must train the v2 model first.")
        print("Starting training will create a new checkpoint compatible with this architecture.")
        return

    try:
        model.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device, weights_only=False))
        print(f"Loaded checkpoint from {IMAGE_MODEL_PATH}")
    except Exception as e:
        print(f"Error: Could not load checkpoint. It is likely an incompatible legacy version.")
        print(f"Error details: {e}")
        print("Please run 'python training/master_train.py' to start a fresh training.")
        return

    model.eval()

    # 4. Load Data
    dataset = ImageDataset(IMAGE_PATH, train=False)
    # Use a small batch size to get exactly 5 images
    dataloader = DataLoader(dataset, batch_size=5, shuffle=True)
    
    # 5. Get 5 images
    batch = next(iter(dataloader))
    images = batch.to(device)

    # 6. Reconstruct
    with torch.no_grad():
        recon_images, _, _ = model(images)

    # 7. Save results
    # Combine original and reconstructed images side-by-side
    # Shape of images/recon_images is [5, 3, 128, 128]
    combined = torch.cat([images, recon_images], dim=0) # [10, 3, 128, 128]
    
    output_path = os.path.join(OUTPUT_DIR, "vae_reconstruction_test.png")
    vutils.save_image(combined, output_path, nrow=5, normalize=True)
    
    print(f"Reconstruction test complete. Result saved to: {output_path}")
    print("Top row: Original Images, Bottom row: Reconstructed Images")

if __name__ == "__main__":
    test_reconstruction()
