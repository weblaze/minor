import sys
import os
import yaml
import wandb
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)
import torch  # Moved to top
import torch.nn as nn  # Moved to top
from torch.utils.data import DataLoader  # Moved to top
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.datasets import AudioFeatureDataset

# Load configuration
config_path = os.path.join(BASE_DIR, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

sys_config = config['system']
audio_config = config['audio_vae']

# Initialize wandb
wandb.init(project="abstraction", name="audio_vae", config=config)

# Paths
AUDIO_FEATURES_PATH = os.path.join(BASE_DIR, config['paths']['audio_features'])
AUDIO_MODEL_PATH = os.path.join(BASE_DIR, config['paths']['models_dir'], "audio_vae.pth")
OUTPUT_DIR = os.path.join(BASE_DIR, config['paths']['models_dir'])  # For saving samples
os.makedirs(OUTPUT_DIR, exist_ok=True)  # Ensure directory exists

# Hyperparameters
LATENT_DIM = sys_config['latent_dim']
BATCH_SIZE = audio_config['batch_size']
NUM_EPOCHS = audio_config['num_epochs']
LEARNING_RATE = float(audio_config['learning_rate'])
BETA = audio_config['beta']
INPUT_CHANNELS = audio_config['input_channels']
TIME_STEPS = audio_config['time_steps']

# Device
device = torch.device(sys_config['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Dataset
try:
    dataset = AudioFeatureDataset(AUDIO_FEATURES_PATH, normalize=True, train=True)
    if len(dataset) == 0:
        raise ValueError("Audio dataset is empty; no .npy files found.")
    print(f"Loaded {len(dataset)} audio feature files.")
except Exception as e:
    raise RuntimeError(f"Failed to load audio dataset: {e}")

dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

# Model
audio_vae = AudioVAE(input_channels=INPUT_CHANNELS, time_steps=TIME_STEPS, latent_dim=LATENT_DIM).to(device)
optimizer = torch.optim.Adam(audio_vae.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

# Loss function
def vae_loss(recon_x, x, mu, logvar, beta=BETA):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    logvar = torch.clamp(logvar, min=-10, max=10)
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recon_loss + beta * kl_div, recon_loss, kl_div

# Training loop
audio_vae.train()
for epoch in range(NUM_EPOCHS):
    total_loss = 0
    total_recon_loss = 0
    total_kl = 0
    
    # Warm-up phase for KL loss
    beta = 0.0 if epoch < 10 else min(BETA, BETA * (epoch - 10 + 1) / 10)
    
    for batch_idx, batch in enumerate(dataloader):
        mfccs = batch['mfccs'].to(device)  # Shape: (batch, 20, 216)
        spectral_centroid = batch['spectral_centroid'].to(device)  # Shape: (batch, 1, 216)
        rms = batch['rms'].to(device)  # Shape: (batch, 1, 216)
        
        audio_features = torch.cat([mfccs, spectral_centroid, rms], dim=1)  # Shape: (batch, 22, 216)

        optimizer.zero_grad()
        recon_audio, mu, logvar = audio_vae(audio_features)
        
        # Debug prints every 5 epochs for first batch
        if batch_idx == 0 and epoch % 5 == 0:
            print(f"Epoch {epoch+1}, Input min/max: {audio_features.min().item():.4f}/{audio_features.max().item():.4f}")
            print(f"Epoch {epoch+1}, Recon min/max: {recon_audio.min().item():.4f}/{recon_audio.max().item():.4f}")
            print(f"Epoch {epoch+1}, mu std: {mu.std().item():.4f}, logvar std: {logvar.std().item():.4f}")

        loss, recon_loss, kl_div = vae_loss(recon_audio, audio_features, mu, logvar, beta=beta)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(audio_vae.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        total_recon_loss += recon_loss.item()
        total_kl += kl_div.item()

        # Save sample at last epoch
        if epoch == NUM_EPOCHS - 1 and batch_idx == 0:
            torch.save(recon_audio.cpu(), os.path.join(OUTPUT_DIR, "sample_recon_audio.pt"))
            torch.save(audio_features.cpu(), os.path.join(OUTPUT_DIR, "sample_orig_audio.pt"))

    scheduler.step()
    avg_loss = total_loss / len(dataloader.dataset)
    avg_recon_loss = total_recon_loss / len(dataloader.dataset)
    avg_kl = total_kl / len(dataloader.dataset)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Loss: {avg_loss:.4f}, Recon Loss: {avg_recon_loss:.4f}, KL: {avg_kl:.4f}")

    wandb.log({
        "epoch": epoch + 1,
        "loss": avg_loss,
        "recon_loss": avg_recon_loss,
        "kl": avg_kl
    })

# Save model
torch.save(audio_vae.state_dict(), AUDIO_MODEL_PATH)
print(f"✅ Saved trained AudioVAE to {AUDIO_MODEL_PATH}")