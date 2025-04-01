# D:\musicc\minor\visualization\visualize_latent_space.py
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# Add the parent directory to the system path
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Import the models
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.datasets import ImageDataset, AudioFeatureDataset
from models.autoencoder.mapping_network import MappingNetwork

# Paths
IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "abstract_art")
AUDIO_PATH = os.path.join(BASE_DIR, "datasets", "audio_features")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "image_vae.pth")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "audio_vae.pth")
MAPPING_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "mapping_network.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, "visualization", "latent_space_plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200, random_state=42)
# Hyperparameters
LATENT_DIM = 256
BATCH_SIZE = 8
NUM_SAMPLES_PER_DOMAIN = 100  # Number of samples to visualize per domain (image and audio)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
image_dataset = ImageDataset(IMAGE_PATH, train=True)
audio_dataset = AudioFeatureDataset(AUDIO_PATH, normalize=True, train=True)

# Debug: Print dataset lengths
print(f"Number of images: {len(image_dataset)}")
print(f"Number of audio features: {len(audio_dataset)}")

# Limit the number of samples per domain
num_image_samples = min(NUM_SAMPLES_PER_DOMAIN, len(image_dataset))
num_audio_samples = min(NUM_SAMPLES_PER_DOMAIN, len(audio_dataset))

if num_image_samples == 0 or num_audio_samples == 0:
    raise ValueError("At least one sample is required from each dataset for visualization.")

# Create subsets for visualization
image_subset = torch.utils.data.Subset(image_dataset, range(num_image_samples))
audio_subset = torch.utils.data.Subset(audio_dataset, range(num_audio_samples))

image_loader = DataLoader(image_subset, batch_size=BATCH_SIZE, shuffle=False)
audio_loader = DataLoader(audio_subset, batch_size=BATCH_SIZE, shuffle=False)

# Load models
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
image_vae.eval()

audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)
audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
audio_vae.eval()

mapping_network = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM).to(device)
mapping_network.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device))
mapping_network.eval()

# Function to extract image latent representations
def extract_image_latents(loader, model, num_samples):
    latents = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * BATCH_SIZE >= num_samples:
                break
            images = batch.to(device)
            mu, logvar = model.encode(images)
            # Clamp logvar to prevent numerical instability
            logvar = torch.clamp(logvar, min=-10, max=10)
            z = model.reparameterize(mu, logvar)
            latents.append(z.cpu().numpy())
    latents = np.concatenate(latents, axis=0)[:num_samples]
    if np.any(np.isnan(latents)) or np.any(np.isinf(latents)):
        print(f"Warning: Found {np.sum(np.isnan(latents))} NaN and {np.sum(np.isinf(latents))} inf values in image latents")
    return latents

# Function to extract audio latent representations
def extract_audio_latents(loader, audio_vae, mapping_network, num_samples):
    latents = []
    with torch.no_grad():
        for i, batch in enumerate(loader):
            if i * BATCH_SIZE >= num_samples:
                break
            audio_features = batch.transpose(1, 2).to(device)  # [batch_size, 20, 216]
            mu_audio, logvar_audio, _ = audio_vae.encode(audio_features)
            # Clamp logvar to prevent numerical instability
            logvar_audio = torch.clamp(logvar_audio, min=-10, max=10)
            z_audio = audio_vae.reparameterize(mu_audio, logvar_audio)  # [batch_size, 256]
            mu_mapped, logvar_mapped = mapping_network(z_audio)
            # Clamp mapped logvar as well
            logvar_mapped = torch.clamp(logvar_mapped, min=-10, max=10)
            z_mapped = mapping_network.reparameterize(mu_mapped, logvar_mapped)  # [batch_size, 256]
            latents.append(z_mapped.cpu().numpy())
    latents = np.concatenate(latents, axis=0)[:num_samples]
    if np.any(np.isnan(latents)) or np.any(np.isinf(latents)):
        print(f"Warning: Found {np.sum(np.isnan(latents))} NaN and {np.sum(np.isinf(latents))} inf values in audio latents")
    return latents

# Extract latent representations
print("Extracting image latents...")
image_latents = extract_image_latents(image_loader, image_vae, num_image_samples)
print("Extracting audio latents...")
audio_latents = extract_audio_latents(audio_loader, audio_vae, mapping_network, num_audio_samples)

# Combine latents for t-SNE
combined_latents = np.concatenate([image_latents, audio_latents], axis=0)
labels = np.array([0] * num_image_samples + [1] * num_audio_samples)  # 0 for images, 1 for audio

# Check for NaN or inf values and handle them
if np.any(np.isnan(combined_latents)) or np.any(np.isinf(combined_latents)):
    print("Found NaN or inf values in combined latents. Filtering out invalid samples...")
    valid_mask = np.isfinite(combined_latents).all(axis=1)
    combined_latents = combined_latents[valid_mask]
    labels = labels[valid_mask]
    num_valid_samples = len(combined_latents)
    print(f"Reduced to {num_valid_samples} valid samples after filtering.")
    if num_valid_samples < 2:
        raise ValueError("Not enough valid samples remaining for t-SNE after filtering NaN/inf values.")

# Apply t-SNE for dimensionality reduction
print("Applying t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(combined_latents) - 1))
latent_2d = tsne.fit_transform(combined_latents)

# Split back into image and audio
image_2d = latent_2d[:num_image_samples]
audio_2d = latent_2d[num_image_samples:num_image_samples + num_audio_samples]

# Plotting
plt.figure(figsize=(10, 8))
plt.scatter(image_2d[:, 0], image_2d[:, 1], c='blue', label='Image Latents', alpha=0.6)
plt.scatter(audio_2d[:, 0], audio_2d[:, 1], c='red', label='Audio Latents', alpha=0.6)

plt.title("Latent Space Distribution (Image vs Audio)")
plt.xlabel("t-SNE Dimension 1")
plt.ylabel("t-SNE Dimension 2")
plt.legend()
plt.grid(True)

# Save the plot
output_path = os.path.join(OUTPUT_DIR, "latent_space_distribution.png")
plt.savefig(output_path)
plt.close()
print(f"✅ Saved latent space visualization to {output_path}")