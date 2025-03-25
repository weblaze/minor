import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.datasets import AudioFeatureDataset
import os
import math

# Paths
AUDIO_FEATURES_PATH = "datasets/audio_features/"
AUDIO_MODEL_PATH = "tmodels/audio_vae.pth"

# Hyperparameters
BATCH_SIZE = 16
EPOCHS = 100
LEARNING_RATE = 1e-4
LATENT_DIM = 64
MAX_BETA = 10.0
RECON_WEIGHT = 0.1
WARMUP_EPOCHS = 5
KLD_LOWER_BOUND = 3.0
KLD_UPPER_BOUND = 6.0
BETA_ADJUST_RATE = 0.5
MIN_KLD = 3.0
CYCLE_LENGTH = 10

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Dataset and Loader
audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=True)
audio_loader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True)

# Model
audio_vae = AudioVAE(input_channels=20, time_steps=216, latent_dim=LATENT_DIM).to(device)  # Updated time_steps

# Optimizer and Scheduler
audio_optimizer = optim.Adam(audio_vae.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
audio_scheduler = ReduceLROnPlateau(audio_optimizer, mode='min', factor=0.5, patience=30)

# Cyclical annealing for KLD weight
def cyclical_annealing(epoch, cycle_length=CYCLE_LENGTH):
    cycle = math.floor(1 + epoch / (2 * cycle_length))
    x = abs(epoch / cycle_length - 2 * cycle + 1)
    tau = max(0, 1 - x)
    return tau

# Loss function with cyclical annealing and minimum KLD constraint
def vae_loss(recon_x, x, mu, logvar, beta, epoch):
    MSE = nn.MSELoss()(recon_x, x)
    logvar = torch.clamp(logvar, min=-10, max=10)
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    kld_weight = cyclical_annealing(epoch)
    kld_loss = beta * kld_weight * KLD
    kld_penalty = 0.0
    if KLD < MIN_KLD:
        kld_penalty = (MIN_KLD - KLD) * 10.0
    return RECON_WEIGHT * MSE + kld_loss + kld_penalty, MSE, KLD

# Linear warmup for beta
def warmup_beta(epoch, warmup_epochs=WARMUP_EPOCHS, max_beta=MAX_BETA, min_beta=0.001):
    if epoch < warmup_epochs:
        return min_beta + (max_beta - min_beta) * (epoch / warmup_epochs)
    return max_beta

# Dynamic beta adjustment based on KLD
def adjust_beta(current_beta, kld, lower_bound=KLD_LOWER_BOUND, upper_bound=KLD_UPPER_BOUND, adjust_rate=BETA_ADJUST_RATE):
    if kld < lower_bound:
        return min(MAX_BETA, current_beta * (1 + adjust_rate))
    elif kld > upper_bound:
        return max(0.001, current_beta * (1 - adjust_rate))
    else:
        return current_beta

# Training Audio VAE
print("Training Audio VAE...")
current_beta = 0.001
for epoch in range(EPOCHS):
    audio_vae.train()
    audio_loss = 0
    mse_loss_total = 0
    kld_loss_total = 0
    base_beta = warmup_beta(epoch)
    for features in audio_loader:
        features = features.transpose(1, 2).to(device)
        audio_optimizer.zero_grad()
        mu, logvar, skips = audio_vae.encode(features)
        z = audio_vae.reparameterize(mu, logvar)
        recon = audio_vae.decode(z, skips)
        loss, mse, kld = vae_loss(recon, features, mu, logvar, current_beta, epoch)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(audio_vae.parameters(), max_norm=10.0)
        audio_optimizer.step()
        audio_loss += loss.item()
        mse_loss_total += mse.item()
        kld_loss_total += kld.item()
    avg_kld = kld_loss_total / len(audio_loader)
    current_beta = adjust_beta(base_beta, avg_kld)
    audio_scheduler.step(audio_loss / len(audio_loader))
    print(f"Epoch [{epoch+1}/{EPOCHS}], Audio Loss: {audio_loss/len(audio_loader):.4f}, MSE: {mse_loss_total/len(audio_loader):.4f}, KLD: {kld_loss_total/len(audio_loader):.4f}, Beta: {current_beta:.4f}")

# Save the model
torch.save(audio_vae.state_dict(), AUDIO_MODEL_PATH)
print(f"✅ Audio VAE model saved to {AUDIO_MODEL_PATH}")