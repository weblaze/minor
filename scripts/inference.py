import os
import torch
import torch.nn as nn
import numpy as np
import laion_clap
from PIL import Image
import yaml

from models.diffusion.unet import ConditionalUNet
from models.diffusion.scheduler import DDPMScheduler
from models.autoencoder.image_vae import ImageVAE

# Load centralized config
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = os.path.join(BASE_DIR, "configs", "config.yaml")
with open(config_path, "r") as f:
    config = yaml.safe_load(f)

LATENT_DIM = config['system']['latent_dim']
DEVICE = torch.device(config['system']['device'] if torch.cuda.is_available() else "cpu")
SPATIAL_C = config['latent_diffusion'].get('latent_channels', 8)
SPATIAL_H = SPATIAL_W = config['latent_diffusion'].get('spatial_size', 16)

def load_models(base_dir):
    """Loads and returns the CLAP module, Diffusion UNet, and ImageVAE."""
    # Weights paths
    CLAP_CKPT = os.path.join(base_dir, "music_audioset_epoch_15_esc_90.14.pt")
    LDM_MODEL_PATH = os.path.join(base_dir, "tmodels", "latent_diffusion.pth")
    IMAGE_MODEL_PATH = os.path.join(base_dir, "tmodels", "image_vae.pth")

    print(f"Loading CLAP on {DEVICE}...")
    clap_model = laion_clap.CLAP_Module(enable_fusion=False, amodel='HTSAT-base')
    clap_model.load_ckpt(CLAP_CKPT)
    clap_model = clap_model.to(DEVICE)
    clap_model.eval()
    print("[LOG] CLAP Loaded.")

    print(f"Loading Image VAE...")
    image_vae = ImageVAE(latent_dim=LATENT_DIM).to(DEVICE)
    if os.path.exists(IMAGE_MODEL_PATH):
        image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=DEVICE))
        print("[LOG] VAE Loaded.")
    image_vae.eval()

    print(f"Loading Diffusion UNet...")
    unet = ConditionalUNet(
        in_channels=SPATIAL_C, 
        out_channels=SPATIAL_C, 
        time_emb_dim=config['latent_diffusion']['time_emb_dim'],
        condition_dim=512
    ).to(DEVICE)
    if os.path.exists(LDM_MODEL_PATH):
        unet.load_state_dict(torch.load(LDM_MODEL_PATH, map_location=DEVICE))
        print("[LOG] UNet Loaded.")
    unet.eval()

    # Scheduler instance for inference
    scheduler = DDPMScheduler(
        num_train_timesteps=config['latent_diffusion']['num_train_timesteps']
    )
    scheduler.set_device(DEVICE)

    return clap_model, unet, image_vae, scheduler

def extract_audio_embedding(clap_model, audio_path):
    """Extracts 512D CLAP embedding from an audio file."""
    with torch.no_grad():
        # get_audio_embedding_from_filelist returns list of embeddings
        embed = clap_model.get_audio_embedding_from_filelist(x=[audio_path], use_tensor=True)
    return embed # [1, 512]

def generate_diffusion(clap_model, unet, image_vae, scheduler, audio_path=None, num_steps=50, progress_callback=None):
    """
    Generates an image from audio using the Latent Diffusion process.
    If audio_path is None, it runs in "Dream Mode" (null conditioning).
    """
    # 1. Prepare Conditioning
    if audio_path:
        print(f"[LOG] Extracting audio embedding for {audio_path}...")
        conditioning = extract_audio_embedding(clap_model, audio_path).to(DEVICE)
        print("[LOG] Conditioning vector extracted.")
    else:
        # Dreaming Mode: Null conditioning (zero vector)
        print("[LOG] Entering Dreaming Mode (Null conditioning).")
        conditioning = torch.zeros((1, 512), device=DEVICE)

    # 2. Sample initial noise [1, 32, 4, 4]
    latents = torch.randn((1, SPATIAL_C, SPATIAL_H, SPATIAL_W), device=DEVICE)
    
    # 3. Setup timesteps for inference (linear spread)
    total_train_steps = config['latent_diffusion']['num_train_timesteps']
    # We work backwards from T to 0
    inference_timesteps = np.linspace(total_train_steps - 1, 0, num_steps).astype(int)
    
    print(f"[LOG] Starting denoising loop for {num_steps} steps...")
    # 4. Denoising Loop
    with torch.no_grad():
        for i, t in enumerate(inference_timesteps):
            t_tensor = torch.tensor([t], device=DEVICE).long()
            
            # Predict noise residual
            noise_pred = unet(latents, t_tensor, conditioning)
            
            # Step scheduler to get previous sample
            latents = scheduler.step(noise_pred, t, latents)
            
            # Progress update
            if progress_callback:
                progress_callback(int((i + 1) / num_steps * 100))
    print("[LOG] Denoising complete.")

    # 5. Decode Latent to Image
    print("[LOG] Decoding latent to pixel space...")
    with torch.no_grad():
        # Flatten 32*4*4 back to 512 for the Linear-based ImageVAE decoder
        z_flat = latents.view(1, -1) 
        output_image = image_vae.decode(z_flat)
        print("[LOG] Decoding complete.")
        
        # Post-process: [-1, 1] to [0, 1]
        output_image = (output_image + 1) / 2
        output_image = torch.clamp(output_image, 0, 1)

    return output_image
