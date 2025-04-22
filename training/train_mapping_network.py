import sys
import os
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.mapping_network import MappingNetwork, InverseMappingNetwork
from models.autoencoder.datasets import AudioFeatureDataset, ImageDataset



# Paths
AUDIO_FEATURES_PATH = os.path.join(BASE_DIR, "datasets", "audio_features")
IMAGE_PATH = os.path.join(BASE_DIR, "datasets", "abstract_art")
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "audio_vae.pth")
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "image_vae.pth")
MAPPING_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "mapping_network.pth")
INVERSE_MAPPING_MODEL_PATH = os.path.join(BASE_DIR, "tmodels", "inverse_mapping_network.pth")
os.makedirs(os.path.dirname(MAPPING_MODEL_PATH), exist_ok=True)

# Hyperparameters
LATENT_DIM = 512  # Matches updated VAEs
BATCH_SIZE = 16  # Reduced from 32 for stability with larger latent dim
NUM_EPOCHS = 50  # Increased from 10, matches older script
LEARNING_RATE = 1e-4
BETA = 0.005  # Reduced from 0.05, inspired by AudioVAE success
CYCLE_WEIGHT = 1.0
DIV_WEIGHT = 0.001  # Reduced from 0.01, gradual warm-up
MMD_WEIGHT = 5.0

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load datasets
audio_dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=True)
image_dataset = ImageDataset(IMAGE_PATH, train=True)
print(f"Loaded {len(audio_dataset)} audio files, {len(image_dataset)} image files.")
audio_dataloader = DataLoader(audio_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
image_dataloader = DataLoader(image_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Load pre-trained VAEs
audio_vae = AudioVAE(input_channels=22, time_steps=216, latent_dim=LATENT_DIM).to(device)
image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
audio_vae.eval()
image_vae.eval()

# Initialize mapping networks with weight initialization
def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias, 0)

mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM, condition_dim=3).to(device)
inverse_mapping_net = InverseMappingNetwork(image_latent_dim=LATENT_DIM, audio_latent_dim=LATENT_DIM).to(device)
mapping_net.apply(init_weights)
inverse_mapping_net.apply(init_weights)

# Optimizers
optimizer_map = torch.optim.Adam(mapping_net.parameters(), lr=LEARNING_RATE)
optimizer_inv = torch.optim.Adam(inverse_mapping_net.parameters(), lr=LEARNING_RATE)
scheduler_map = torch.optim.lr_scheduler.StepLR(optimizer_map, step_size=20, gamma=0.5)
scheduler_inv = torch.optim.lr_scheduler.StepLR(optimizer_inv, step_size=20, gamma=0.5)

# Loss functions
def kl_divergence(mu, logvar):
    logvar = torch.clamp(logvar, min=-10, max=10)
    return -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())

def cycle_consistency_loss(original, reconstructed):
    return nn.functional.mse_loss(reconstructed, original, reduction='mean')

def diversity_loss(z1, z2):
    return -torch.mean(torch.norm(z1 - z2, dim=1))

def compute_mmd(x, y, sigma=None):
    def rbf_kernel(x1, x2, sigma):
        x1_norm = torch.sum(x1 ** 2, dim=1).view(-1, 1)
        x2_norm = torch.sum(x2 ** 2, dim=1).view(1, -1)
        gamma = 1.0 / (2 * sigma ** 2)
        K = torch.exp(-gamma * (x1_norm + x2_norm - 2 * torch.matmul(x1, x2.t())))
        return K

    if sigma is None:
        xy = torch.cat([x, y], dim=0)
        dists = torch.pdist(xy)
        sigma = torch.median(dists) if dists.numel() > 0 else 1.0
        sigma = sigma.clamp(min=1e-3)

    Kxx = rbf_kernel(x, x, sigma)
    Kyy = rbf_kernel(y, y, sigma)
    Kxy = rbf_kernel(x, y, sigma)
    mmd = torch.mean(Kxx) + torch.mean(Kyy) - 2 * torch.mean(Kxy)
    return mmd.clamp(min=0)

# Training loop
mapping_net.train()
inverse_mapping_net.train()
image_iter = iter(image_dataloader)

for epoch in range(NUM_EPOCHS):
    total_loss_map, total_loss_inv = 0, 0
    total_kl_map, total_kl_inv = 0, 0
    total_cycle_audio, total_cycle_image = 0, 0
    total_div, total_mmd = 0, 0
    batches_processed = 0

    # Warm-up phase for diversity and KL
    div_weight = min(DIV_WEIGHT, DIV_WEIGHT * (epoch + 1) / 10) if epoch < 10 else DIV_WEIGHT
    beta = 0.0 if epoch < 10 else min(BETA, BETA * (epoch - 10 + 1) / 10)

    for audio_batch in audio_dataloader:
        try:
            image_batch = next(image_iter)
        except StopIteration:
            image_iter = iter(image_dataloader)
            image_batch = next(image_iter)

        # Prepare audio features and condition
        mfccs = audio_batch['mfccs'].to(device)  # [batch, 20, 216]
        spectral_centroid = audio_batch['spectral_centroid'].to(device)  # [batch, 1, 216]
        rms = audio_batch['rms'].to(device)  # [batch, 1, 216]
        audio_features = torch.cat([mfccs, spectral_centroid, rms], dim=1)  # [batch, 22, 216]
        condition = torch.stack([
            spectral_centroid.mean(dim=2).squeeze(1),
            rms.mean(dim=2).squeeze(1),
            audio_batch['tempo'].squeeze().to(device)  # [batch,]
        ], dim=1)  # [batch, 3]
        images = image_batch.to(device)

        # Encode to latent space
        with torch.no_grad():
            mu_audio, logvar_audio, _ = audio_vae.encode(audio_features)
            z_audio = audio_vae.reparameterize(mu_audio, logvar_audio)
            mu_image, logvar_image, _ = image_vae.encode(images)
            z_image = image_vae.reparameterize(mu_image, logvar_image)

        # Forward mapping: audio → image
        optimizer_map.zero_grad()
        mapped_mu, mapped_logvar = mapping_net(z_audio, condition)
        z_mapped = mapping_net.reparameterize(mapped_mu, mapped_logvar)

        # Inverse mapping: image → audio
        optimizer_inv.zero_grad()
        inv_mu, inv_logvar = inverse_mapping_net(z_image)
        z_inv = inverse_mapping_net.reparameterize(inv_mu, inv_logvar)

        # Cycle consistency
        cycled_mu_audio, cycled_logvar_audio = inverse_mapping_net(z_mapped)
        z_cycled_audio = inverse_mapping_net.reparameterize(cycled_mu_audio, cycled_logvar_audio)
        cycled_mu_image, cycled_logvar_image = mapping_net(z_inv, condition)
        z_cycled_image = mapping_net.reparameterize(cycled_mu_image, cycled_logvar_image)

        # Losses for forward mapping
        kl_loss_map = kl_divergence(mapped_mu, mapped_logvar)
        cycle_loss_audio = cycle_consistency_loss(z_audio, z_cycled_audio)
        half = BATCH_SIZE // 2
        z1, z2 = mapped_mu[:half], mapped_mu[half:]
        div_loss = diversity_loss(z1, z2) if half > 0 else 0
        mmd_loss = compute_mmd(z_mapped, z_image)

        # Losses for inverse mapping
        kl_loss_inv = kl_divergence(inv_mu, inv_logvar)
        cycle_loss_image = cycle_consistency_loss(z_image, z_cycled_image)

        # Total losses
        loss_map = beta * kl_loss_map + CYCLE_WEIGHT * cycle_loss_audio + div_weight * div_loss + MMD_WEIGHT * mmd_loss
        loss_inv = beta * kl_loss_inv + CYCLE_WEIGHT * cycle_loss_image

        # Backward pass
        loss_map.backward()
        loss_inv.backward()
        torch.nn.utils.clip_grad_norm_(mapping_net.parameters(), max_norm=0.5)
        torch.nn.utils.clip_grad_norm_(inverse_mapping_net.parameters(), max_norm=0.5)
        optimizer_map.step()
        optimizer_inv.step()

        # Accumulate losses
        total_loss_map += loss_map.item()
        total_loss_inv += loss_inv.item()
        total_kl_map += kl_loss_map.item()
        total_kl_inv += kl_loss_inv.item()
        total_cycle_audio += cycle_loss_audio.item()
        total_cycle_image += cycle_loss_image.item()
        total_div += div_loss.item()
        total_mmd += mmd_loss.item()
        batches_processed += 1

        # Debug print every 10 epochs for first batch
        if batches_processed == 1 and epoch % 10 == 0:
            print(f"Epoch {epoch+1}, z_audio std: {z_audio.std().item():.4f}, z_image std: {z_image.std().item():.4f}")
            print(f"Epoch {epoch+1}, z_mapped std: {z_mapped.std().item():.4f}, z_inv std: {z_inv.std().item():.4f}")

    scheduler_map.step()
    scheduler_inv.step()

    if batches_processed > 0:
        avg_loss_map = total_loss_map / batches_processed
        avg_loss_inv = total_loss_inv / batches_processed
        avg_kl_map = total_kl_map / batches_processed
        avg_kl_inv = total_kl_inv / batches_processed
        avg_cycle_audio = total_cycle_audio / batches_processed
        avg_cycle_image = total_cycle_image / batches_processed
        avg_div = total_div / batches_processed
        avg_mmd = total_mmd / batches_processed
    else:
        avg_loss_map = avg_loss_inv = avg_kl_map = avg_kl_inv = avg_cycle_audio = avg_cycle_image = avg_div = avg_mmd = float('nan')

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, "
          f"Map Loss: {avg_loss_map:.4f}, KL Map: {avg_kl_map:.4f}, Cycle Audio: {avg_cycle_audio:.4f}, Div: {avg_div:.4f}, MMD: {avg_mmd:.4f}, "
          f"Inv Loss: {avg_loss_inv:.4f}, KL Inv: {avg_kl_inv:.4f}, Cycle Image: {avg_cycle_image:.4f}, "
          f"Batches Processed: {batches_processed}")

# Save the models
torch.save(mapping_net.state_dict(), MAPPING_MODEL_PATH)
torch.save(inverse_mapping_net.state_dict(), INVERSE_MAPPING_MODEL_PATH)
print(f"✅ Saved trained MappingNetwork to {MAPPING_MODEL_PATH}")
print(f"✅ Saved trained InverseMappingNetwork to {INVERSE_MAPPING_MODEL_PATH}")