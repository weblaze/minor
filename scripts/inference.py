import os
import torch
import librosa
import numpy as np

from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.image_vae import ImageVAE
from models.autoencoder.mapping_network import MappingNetwork

LATENT_DIM = 512
CHANNELS = 22
TIME_STEPS = 216
LOGVAR_SCALE = 1.0
OUTPUT_SCALE = 1.0

# Using CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_models(base_dir):
    """Loads and returns the AudioVAE, ImageVAE, and MappingNetwork models."""
    AUDIO_MODEL_PATH = os.path.join(base_dir, "tmodels", "audio_vae.pth")
    IMAGE_MODEL_PATH = os.path.join(base_dir, "tmodels", "image_vae.pth")
    MAPPING_MODEL_PATH = os.path.join(base_dir, "tmodels", "mapping_network.pth")

    audio_vae = AudioVAE(input_channels=CHANNELS, time_steps=TIME_STEPS, latent_dim=LATENT_DIM).to(device)
    image_vae = ImageVAE(latent_dim=LATENT_DIM).to(device)
    mapping_net = MappingNetwork(audio_latent_dim=LATENT_DIM, image_latent_dim=LATENT_DIM, condition_dim=3).to(device)

    try:
        audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device))
        image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device))
        mapping_net.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device))
    except Exception as e:
        print(f"Error loading weights: {e}")
        print("Falling back to partial loading...")
        audio_vae.load_state_dict(torch.load(AUDIO_MODEL_PATH, map_location=device), strict=False)
        image_vae.load_state_dict(torch.load(IMAGE_MODEL_PATH, map_location=device), strict=False)
        mapping_net.load_state_dict(torch.load(MAPPING_MODEL_PATH, map_location=device), strict=False)

    audio_vae.eval()
    image_vae.eval()
    mapping_net.eval()
    
    return audio_vae, image_vae, mapping_net

def extract_audio_features(mp3_path_or_bytes, sr=22050, n_mfcc=20, hop_length=512, target_time_steps=216):
    """Extracts features from an audio file and formats them for the model."""
    audio, sr = librosa.load(mp3_path_or_bytes, sr=sr)
    
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)
    spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=audio, hop_length=hop_length)
    tempo, _ = librosa.beat.beat_track(y=audio, sr=sr, hop_length=hop_length)
    
    current_time_steps = mfccs.shape[1]
    if current_time_steps > target_time_steps:
        mfccs = mfccs[:, :target_time_steps]
        spectral_centroid = spectral_centroid[:, :target_time_steps]
        rms = rms[:, :target_time_steps]
    elif current_time_steps < target_time_steps:
        pad_width = target_time_steps - current_time_steps
        mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        spectral_centroid = np.pad(spectral_centroid, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        rms = np.pad(rms, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
    
    mfccs = (mfccs - np.mean(mfccs)) / (np.std(mfccs) + 1e-8)
    spectral_centroid = spectral_centroid / 4000.0
    rms = (rms - np.mean(rms)) / (np.std(rms) + 1e-8)
    tempo = (tempo - 60) / (200 - 60)
    
    mfccs_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0)
    spectral_centroid_tensor = torch.tensor(spectral_centroid, dtype=torch.float32).unsqueeze(0)
    rms_tensor = torch.tensor(rms, dtype=torch.float32).unsqueeze(0)
    tempo_tensor = torch.tensor(np.array([tempo]), dtype=torch.float32)
    
    audio_features = torch.cat([mfccs_tensor, spectral_centroid_tensor, rms_tensor], dim=1)
    condition = torch.stack([
        spectral_centroid_tensor.mean(dim=2).squeeze(1),
        rms_tensor.mean(dim=2).squeeze(1),
        tempo_tensor.squeeze(0)
    ], dim=1)
    
    return audio_features, condition

def generate_image_from_audio(audio_features, condition, audio_vae, mapping_net, image_vae):
    """Passes audio features through the pipeline to generate an image."""
    audio_features = audio_features.to(device)
    condition = condition.to(device)
    
    with torch.no_grad():
        mu_audio, logvar_audio, _ = audio_vae.encode(audio_features)
        z_audio = audio_vae.reparameterize(mu_audio, logvar_audio)
    
    with torch.no_grad():
        mapped_mu, mapped_logvar = mapping_net(z_audio, condition)
        mapped_logvar = torch.clamp(mapped_logvar * LOGVAR_SCALE, min=-10, max=10)
        z_image = mapping_net.reparameterize(mapped_mu, mapped_logvar)
    
    with torch.no_grad():
        generated_image = image_vae.decode(z_image)
        generated_image = torch.clamp(generated_image * OUTPUT_SCALE, -1, 1)
    
    if torch.isnan(generated_image).any() or torch.isinf(generated_image).any():
        generated_image = torch.nan_to_num(generated_image, nan=0.0, posinf=1.0, neginf=-1.0)
    
    # Scale to [0, 1] for image saving/display
    generated_image = (generated_image + 1) / 2
    return generated_image
