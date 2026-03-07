import sys
import os
import yaml
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy as np

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(BASE_DIR)

from models.diffusion.unet import ConditionalUNet
from models.diffusion.scheduler import DDPMScheduler
from models.autoencoder.image_vae import ImageVAE

# Load configuration
config_path = os.path.join(BASE_DIR, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

# Initialize wandb
wandb.init(project="abstraction", name="latent_diffusion", config=config)

sys_config = config['system']
# Add latent diffusion config defaults if missing
ldm_config = config.get('latent_diffusion', {
    'batch_size': 32,
    'num_epochs': 500,
    'learning_rate': 1e-4,
    'time_emb_dim': 128,
    'num_train_timesteps': 1000
})

device = torch.device(sys_config['device'] if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Paths
AUDIO_FEATURES_DIR = os.path.join(BASE_DIR, "datasets", "clap_features")
IMAGE_FEATURES_DIR = os.path.join(BASE_DIR, "datasets", "fma_small_images") # Or your matched image dir
IMAGE_MODEL_PATH = os.path.join(BASE_DIR, config['paths']['models_dir'], "image_vae.pth")
LDM_MODEL_PATH = os.path.join(BASE_DIR, config['paths']['models_dir'], "latent_diffusion.pth")
os.makedirs(os.path.dirname(LDM_MODEL_PATH), exist_ok=True)

class PairedDataset(Dataset):
    """
    Loads paired CLAP embeddings and raw images. 
    Assumes filenames match (e.g., 000002_clap.npy and 000002.jpg).
    For now, simulating behavior if images aren't paired strictly.
    """
    def __init__(self, audio_dir, image_dir):
        self.audio_dir = audio_dir
        self.image_dir = image_dir
        
        # In a real scenario, you'd match the 00002_clap.npy to 00002.jpg
        # Since this path is WIP, we'll list audio features and just grab random images for the structural test
        self.audio_files = [f for f in os.listdir(audio_dir) if f.endswith('_clap.npy')]
        
        # If no image dir provided or exists, we will raise an error soon, but for structure we continue
        from torchvision import transforms
        from models.autoencoder.datasets import ImageDataset
        # Try to use standard dataset if generic image loop
        try:
            self.image_dataset = ImageDataset(os.path.join(BASE_DIR, config['paths']['image_features']), train=True)
            self.image_files = self.image_dataset.image_files
        except:
             self.image_dataset = None

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_dir, self.audio_files[idx])
        # LAION_CLAP is (512,)
        audio_emb = np.load(audio_path)
        audio_emb = torch.tensor(audio_emb, dtype=torch.float32)

        # Grab a corresponding or random image 
        if self.image_dataset is not None:
             image_idx = idx % len(self.image_dataset)
             image = self.image_dataset[image_idx]
        else:
             # Dummy 128x128 image if testing purely audio pipeline
             image = torch.randn(3, 128, 128)
             
        return {'audio_emb': audio_emb, 'image': image}


print("Loading Datasets...")
dataset = PairedDataset(AUDIO_FEATURES_DIR, IMAGE_FEATURES_DIR)
dataloader = DataLoader(dataset, batch_size=ldm_config['batch_size'], shuffle=True, drop_last=True)
print(f"Loaded {len(dataset)} paired samples.")

print("Loading Image VAE...")
# Freeze Image VAE and keep in eval mode. We ONLY train the UNet.
image_vae = ImageVAE(latent_dim=sys_config['latent_dim']).to(device)
try:
    image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
    print("✅ Loaded Pre-trained Image VAE weights!")
except FileNotFoundError:
    print("⚠️ Pre-trained Image VAE not found. Using untrained weights for structural test.")

image_vae.eval()
for param in image_vae.parameters():
    param.requires_grad = False

print("Initializing Diffusion UNet...")
# The latent dim is the spatial dimension output by VAE encoder. 
# Previously trained ImageVAE encodes 3x128x128 -> 512x4x4. 
# Wait, let's check ImageVAE `encode` output: `h.view(h.size(0), -1)` -> FC to Latent Dim (512 flat).
# For diffusion, spatial latent grids are better (e.g. 1024x4x4). 
# We need to diffuse the *spatial grid* before flattening (or rebuild a Conv VAE).
# To keep it compatible with the previous code's 1024x4x4 linear projection, 
# for THIS specific diffusion test, we will diffuse a flattened 512D vector, requiring a 1D UNet or MLP,
# OR we reshape it back into a spatial grid. 
# Let's reshape 512 -> 32x4x4 for a mini 2D UNet to diffuse.
SPATIAL_C, SPATIAL_H, SPATIAL_W = 32, 4, 4 

unet = ConditionalUNet(
    in_channels=SPATIAL_C, 
    out_channels=SPATIAL_C, 
    time_emb_dim=ldm_config['time_emb_dim'],
    condition_dim=512 # CLAP dim
).to(device)

optimizer = torch.optim.Adam(unet.parameters(), lr=float(ldm_config['learning_rate']))
scheduler_noise = DDPMScheduler(num_train_timesteps=ldm_config['num_train_timesteps'])
scheduler_noise.set_device(device)

def train_loop():
    unet.train()
    epochs = ldm_config['num_epochs']
    
    for epoch in range(epochs):
        epoch_loss = 0
        for batch_idx, batch in enumerate(dataloader):
            images = batch['image'].to(device)
            audio_embs = batch['audio_emb'].to(device)
            
            # 1. Encode image to latent space (no gradients)
            with torch.no_grad():
                # The Image VAE outputs mu, logvar for the 512 flat dim
                mu, logvar, _ = image_vae.encode(images)
                z = image_vae.reparameterize(mu, logvar) # [Batch, 512]
            
            # 2. Reshape [Batch, 512] into [Batch, 32, 4, 4] for the spatial UNet
            z_spatial = z.view(-1, SPATIAL_C, SPATIAL_H, SPATIAL_W)
            
            # 3. Sample random noise to add to the latents
            noise = torch.randn_like(z_spatial)
            bsz = z_spatial.shape[0]
            
            # 4. Sample uniform random timesteps for each image in the batch
            timesteps = torch.randint(0, scheduler_noise.num_train_timesteps, (bsz,), device=device).long()
            
            # 5. Add noise to the latents according to the noise magnitude at each timestep (Forward Diffusion)
            noisy_latents = scheduler_noise.add_noise(z_spatial, noise, timesteps)
            
            # 6. Predict the noise residual (Epsilon Prediction)
            optimizer.zero_grad()
            noise_pred = unet(noisy_latents, timesteps, audio_embs)
            
            # 7. Calculate loss (MSE between predicted noise and actual noise added)
            loss = nn.functional.mse_loss(noise_pred, noise)
            loss.backward()
            
            # 8. Optimize
            torch.nn.utils.clip_grad_norm_(unet.parameters(), 1.0)
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if batch_idx == 0 and epoch % 10 == 0:
                 print(f"Epoch {epoch}/{epochs} | Step Loss: {loss.item():.4f}")
                 
        avg_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{epochs} | Avg Loss: {avg_loss:.4f}")
        wandb.log({"epoch": epoch, "loss": avg_loss})
        
    # Save the model
    torch.save(unet.state_dict(), LDM_MODEL_PATH)
    print(f"✅ Saved trained Latent Diffusion UNet to {LDM_MODEL_PATH}")

if __name__ == "__main__":
    if len(dataset) == 0:
        print("⚠️ Waiting for extracted CLAP features (.npy) to begin training.")
        print("Please run `python preprocessing/extract_clap_features.py` natively first.")
    else:
        print("Starting Latent Diffusion Training...")
        train_loop()
