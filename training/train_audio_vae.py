# train_audio_vae.py
import sys
import os
# Dynamically set the base directory
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

# Paths
AUDIO_FEATURES_PATH = os.path.join(BASE_DIR, "datasets", "audio_features")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "audio_vae.pth")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.datasets import AudioFeatureDataset



# Hyperparameters
LATENT_DIM = 256  # Increased for more variety
BATCH_SIZE = 16
NUM_EPOCHS = 50  # Adjust based on dataset size
LEARNING_RATE = 1e-3
BETA = 1.0  # Weight for KL divergence (tune for balance)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset
try:
    dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=True)
    if len(dataset) == 0:
        raise ValueError("Audio dataset is empty; no .npy files found.")
    print(f"Loaded {len(dataset)} audio feature files.")
except Exception as e:
    raise RuntimeError(f"Failed to load audio dataset: {e}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(audio_vae.parameters(), lr=LEARNING_RATE)

# Loss function (VAE loss = reconstruction + KL divergence)
def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div

# Training loop
audio_vae.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    for batch in dataloader:
        audio_features = batch.transpose(1, 2).to(device)  # Shape: (batch, channels, time)
        optimizer.zero_grad()
        recon_audio, mu, logvar = audio_vae(audio_features)
        loss = vae_loss(recon_audio, audio_features, mu, logvar)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_loss = total_loss / len(dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}")

# Save model
torch.save(audio_vae.state_dict(), AUDIO_MODEL_PATH)
print(f"✅ Saved trained AudioVAE to {AUDIO_MODEL_PATH}")