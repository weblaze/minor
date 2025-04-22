import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Paths
AUDIO_FEATURES_PATH = os.path.join(BASE_DIR, "datasets", "audio_features")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "audio_vae.pth")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "image_vae.pth")
MAPPING_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "mapping_network.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "datasets", "generated_images")

import torch
from torch.utils.data import DataLoader
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.mapping_network import MappingNetwork
from models.autoencoder.datasets import AudioFeatureDataset
import torchvision.utils as vutils
import random

# Hyperparameters
LATENT_DIM = 512
BATCH_SIZE = 8
NUM_IMAGES = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create output directory
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load dataset
try:
    audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=False)
    if len(audio_dataset) == 0:
        raise ValueError("Audio dataset is empty; no .npy files found.")
    print(f"Loaded {len(audio_dataset)} audio feature files.")
except Exception as e:
    raise RuntimeError(f"Failed to load audio dataset: {e}")

# Initialize models
audio_vae = AudioVAE(input_channels=22, time_steps=216, latent_dim=LATENT_DIM).to(device)
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM, condition_dim=3).to(device)

# Load model weights
try:
    audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
    image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    mapping_net.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device))
except RuntimeError as e:
    print(f"Error loading model weights: {e}")
    sys.exit(1)

audio_vae.eval()
image_vae.eval()
mapping_net.eval()

# Select random audio indices
audio_indices = random.sample(range(len(audio_dataset)), min(NUM_IMAGES, len(audio_dataset)))
print(f"Selected audio indices: {audio_indices}")

# Generate images
print(f"Generating {NUM_IMAGES} images...")
for i, idx in enumerate(audio_indices):
    with torch.no_grad():
        audio_features_dict = audio_dataset[idx]
        mfccs = audio_features_dict['mfccs'].unsqueeze(0).to(device)  # Shape: [1, 20, 216]
        spectral_centroid = audio_features_dict['spectral_centroid'].unsqueeze(0).to(device)  # Shape: [1, 1, 216]
        rms = audio_features_dict['rms'].unsqueeze(0).to(device)  # Shape: [1, 1, 216]
        tempo = audio_features_dict['tempo'].squeeze().unsqueeze(0).to(device)  # Shape: [1]
        
        audio_features = torch.cat([mfccs, spectral_centroid, rms], dim=1)  # Shape: [1, 22, 216]
        condition = torch.stack([
            spectral_centroid.mean(dim=2).squeeze(1),  # Shape: [1]
            rms.mean(dim=2).squeeze(1),  # Shape: [1]
            tempo  # Shape: [1]
        ], dim=1)  # Shape: [1, 3]
        
        # Encode audio to latent space
        mu_audio, logvar_audio, _ = audio_vae.encode(audio_features)
        z_audio = audio_vae.reparameterize(mu_audio, logvar_audio)  # Shape: [1, 512]
        
        # Map to image latent space
        mapped_mu, mapped_logvar = mapping_net(z_audio, condition)
        z_image = mapping_net.reparameterize(mapped_mu, mapped_logvar)  # Shape: [1, 512]
        
        # Decode to image
        generated_image = image_vae.decode(z_image)  # Shape: [1, 3, 128, 128]
        
        # Save the image
        output_path = os.path.join(OUTPUT_DIR, f"generated_image_{i+1}.png")
        vutils.save_image(generated_image, output_path, normalize=True)
        print(f"Saved image {i+1} to {output_path}")

print(f"✅ Generated {NUM_IMAGES} images in {OUTPUT_DIR}!")