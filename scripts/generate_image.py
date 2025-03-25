import torch
from torch.utils.data import DataLoader
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.mapping_network import MappingNetwork
from models.autoencoder.datasets import AudioFeatureDataset, ImageDataset
import torchvision.utils as vutils
import torchvision.transforms as transforms
import os
import numpy as np
import random  # Added for random sampling

# Paths
AUDIO_FEATURES_PATH = "datasets/audio_features/"
IMAGE_PATH = "datasets/abstract_art/"
AUDIO_MODEL_PATH = "tmodels/audio_vae.pth"
IMAGE_MODEL_PATH = "tomdels/image_vae.pth"
MAPPING_MODEL_PATH = "tmodels/mapping_network.pth"
OUTPUT_DIR = "datasets/generated_images/"

# Hyperparameters
BATCH_SIZE = 16
LATENT_DIM = 64
NUM_IMAGES = 10  # Number of images to generate
LOGVAR_SCALE = 2.0  # Increased from 1.5 to 2.0
NOISE_STD = 0.1  # Standard deviation for random noise

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Create output directory if it doesn't exist
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# Datasets
audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=False)
image_dataset = ImageDataset(IMAGE_PATH, train=False)
image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Select 10 random indices from the audio dataset
audio_indices = random.sample(range(len(audio_dataset)), NUM_IMAGES)
print(f"Selected random audio indices: {audio_indices}")

# Models
audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM).to(device)

# Load pre-trained models
audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH))
image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH))
mapping_net.load_state_dict(torch.load(MAPPING_MODEL_PATH))
audio_vae.eval()
image_vae.eval()
mapping_net.eval()

# Generate 10 Images from Music
print(f"Generating {NUM_IMAGES} images from music...")
for i, idx in enumerate(audio_indices):
    with torch.no_grad():
        # Select a random audio sample using the chosen index
        sample_audio = audio_dataset[idx]
        audio_features = sample_audio.unsqueeze(0).transpose(1, 2).to(device)
        
        # Log basic audio features (mean, std, and range)
        audio_features_np = audio_features.cpu().numpy()
        audio_mean = np.mean(audio_features_np)
        audio_std = np.std(audio_features_np)
        audio_min = np.min(audio_features_np)
        audio_max = np.max(audio_features_np)
        print(f"Image {i+1} (index {idx}) - Audio Features: mean={audio_mean:.4f}, std={audio_std:.4f}, min={audio_min:.4f}, max={audio_max:.4f}")

        # Encode audio to latent space
        mu, logvar, _ = audio_vae.encode(audio_features)
        z_audio = audio_vae.reparameterize(mu, logvar)
        print(f"Image {i+1} (index {idx}) - z_audio min: {z_audio.min().item():.4f}, max: {z_audio.max().item():.4f}")

        # Map audio latent to image latent
        z_image_mu, z_image_logvar = mapping_net(z_audio)
        # Scale logvar to increase variability
        z_image_logvar = z_image_logvar * LOGVAR_SCALE
        z_image = mapping_net.reparameterize(z_image_mu, z_image_logvar)
        # Add random noise to z_image
        noise = torch.randn_like(z_image) * NOISE_STD
        z_image = z_image + noise
        print(f"Image {i+1} (index {idx}) - z_image min: {z_image.min().item():.4f}, max: {z_image.max().item():.4f}")

        # Decode to generate image
        generated_image = image_vae.decode(z_image, [None] * len(image_vae.decoder_layers))
        # Upscale to 512x512
        upscale = transforms.Resize((512, 512))
        generated_image = upscale(generated_image)
        # Save the generated image
        output_path = os.path.join(OUTPUT_DIR, f"generated_image_{i+1}.png")
        vutils.save_image(generated_image, output_path, normalize=True)
        print(f"Saved image {i+1} to {output_path}")

print(f"✅ Generated {NUM_IMAGES} test images in {OUTPUT_DIR}!")