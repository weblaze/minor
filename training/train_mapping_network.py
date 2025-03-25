import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.image_vae import ImageVAE
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder.mapping_network import MappingNetwork
from datasets import AudioFeatureDataset, ImageDataset
import os

# Paths
AUDIO_FEATURES_PATH = "datasets/audio_features/"
IMAGE_PATH = "datasets/abstract_art/"
AUDIO_MODEL_PATH = "tmodels/audio_vae.pth"
IMAGE_MODEL_PATH = "tmodels/image_vae.pth"
MAPPING_MODEL_PATH = "tmodels/mapping_network.pth"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
LATENT_DIM = 64

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Datasets and Loaders
audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=True)
image_dataset = ImageDataset(IMAGE_PATH, train=True)
audio_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True)
image_loader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Models
audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)  # Updated time_steps
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM).to(device)
# Reverse mapping network (image → audio)
reverse_mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM).to(device)

# Load pre-trained models
audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH))
image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH))
audio_vae.eval()
image_vae.eval()

# Optimizer and Scheduler
mapping_optimizer = optim.Adam(list(mapping_net.parameters()) + list(reverse_mapping_net.parameters()), lr=LEARNING_RATE, weight_decay=1e-5)
mapping_scheduler = ReduceLROnPlateau(mapping_optimizer, mode='min', factor=0.5, patience=30)

# Collect image latent statistics
print("Collecting image latent statistics...")
image_vae.eval()
image_latents = []
with torch.no_grad():
    for images in image_loader:
        images = images.to(device)
        mu, logvar, _ = image_vae.encode(images)
        z = image_vae.reparameterize(mu, logvar)
        image_latents.append(z)
image_latents = torch.cat(image_latents, dim=0)
image_mean = torch.mean(image_latents, dim=0)
image_std = torch.std(image_latents, dim=0) + 1e-6

# Loss function with cycle-consistency
def mapping_loss(z_image_pred, target_mean, target_std, z_image_mu, z_image_logvar, z_audio, z_audio_cycle):
    # Distribution matching loss
    pred_mean = torch.mean(z_image_pred, dim=0)
    pred_std = torch.std(z_image_pred, dim=0) + 1e-6
    dist_loss = nn.MSELoss()(pred_mean, target_mean) + nn.MSELoss()(pred_std, target_std)
    # KLD regularization
    kld = -0.5 * torch.sum(1 + z_image_logvar - z_image_mu.pow(2) - z_image_logvar.exp())
    # Cycle-consistency loss
    cycle_loss = nn.MSELoss()(z_audio, z_audio_cycle)
    return dist_loss + 0.1 * kld + 0.5 * cycle_loss

# Training Mapping Network
print("Training Mapping Network...")
mapping_net.train()
reverse_mapping_net.train()
for epoch in range(EPOCHS):
    total_mapping_loss = 0
    audio_iter = iter(audio_loader)
    for features in audio_loader:
        try:
            features = features.transpose(1, 2).to(device)
            with torch.no_grad():
                mu, logvar, _ = audio_vae.encode(features)
                z_audio = audio_vae.reparameterize(mu, logvar)
            mapping_optimizer.zero_grad()
            # Forward mapping: audio → image
            z_image_mu, z_image_logvar = mapping_net(z_audio)
            z_image_pred = mapping_net.reparameterize(z_image_mu, z_image_logvar)
            # Reverse mapping: image → audio
            z_audio_mu_cycle, z_audio_logvar_cycle = reverse_mapping_net(z_image_pred)
            z_audio_cycle = reverse_mapping_net.reparameterize(z_audio_mu_cycle, z_audio_logvar_cycle)
            # Compute loss
            loss = mapping_loss(z_image_pred, image_mean, image_std, z_image_mu, z_image_logvar, z_audio, z_audio_cycle)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=10.0)
            torch.nn.utils.clip_grad_norm_(reverse_mapping_net.parameters(), max_norm=10.0)
            mapping_optimizer.step()
            total_mapping_loss += loss.item()
        except StopIteration:
            audio_iter = iter(audio_loader)
    mapping_scheduler.step(total_mapping_loss / len(audio_loader))
    print(f"Epoch [{epoch+1}/{EPOCHS}], Mapping Loss: {total_mapping_loss/len(audio_loader):.4f}")

torch.save(mapping_net.state_dict(), MAPPING_MODEL_PATH)
print(f"Mapping Network saved to {MAPPING_MODEL_PATH}")