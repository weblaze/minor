# D:\musicc\minor\scripts\generate_image.py
import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

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
import torchvision.transforms as transforms
import numpy as np
import random

LATENT_DIM = 256
BATCH_SIZE = 8
NUM_IMAGES = 10
NOISE_STD = 0.5  # Reduced noise for more consistent generation

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

os.makedirs(OUTPUT_DIR, exist_ok=True)

audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=False)
dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True)

audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM).to(device)

audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
mapping_net.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device))
audio_vae.eval()
image_vae.eval()
mapping_net.eval()

audio_indices = random.sample(range(len(audio_dataset)), min(NUM_IMAGES, len(audio_dataset)))
print(f"Selected audio indices: {audio_indices}")

print(f"Generating {NUM_IMAGES} images...")
for i, idx in enumerate(audio_indices):
    with torch.no_grad():
        audio_features = audio_dataset[idx].unsqueeze(0).transpose(1, 2).to(device)  # [1, 20, 216]
        
        # Encode audio to latent space
        mu_audio, logvar_audio, _ = audio_vae.encode(audio_features)
        z_audio = audio_vae.reparameterize(mu_audio, logvar_audio)  # [1, 256]

        # Map to image latent space
        mapped_mu, mapped_logvar = mapping_net(z_audio)
        z_image = mapping_net.reparameterize(mapped_mu, mapped_logvar)

        # Add a small amount of noise for variation
        z_image += torch.randn_like(z_image) * NOISE_STD

        # Decode to image
        generated_image = image_vae.decode(z_image)  # [1, 3, 128, 128]

        # Upscale to 512x512
        upscale = transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BICUBIC)
        generated_image = upscale(generated_image)

        # Clamp the image to ensure valid pixel values
        generated_image = torch.clamp(generated_image, -1, 1)

        # Save the image
        output_path = os.path.join(OUTPUT_DIR, f"generated_image_{i+1}.png")
        vutils.save_image(generated_image, output_path, normalize=True)
        print(f"Saved image {i+1} to {output_path}")

print(f"✅ Generated {NUM_IMAGES} images in {OUTPUT_DIR}!")