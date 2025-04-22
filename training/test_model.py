# D:\musicc\minor\training\test_model.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import torch
from models.autoencoder.audio_vae import AudioVAE
from models.autoencoder.datasets import AudioFeatureDataset
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = AudioVAE().to(device)
model.load_state_dict(torch.load("D:\\musicc\\minor\\tmodels\\audio_vae.pth"))
model.eval()

dataset = AudioFeatureDataset("D:\\musicc\\minor\\datasets\\audio_features")
num_samples = min(10, len(dataset))  # Test 10 samples or fewer if dataset is smaller
total_mse = 0

for i in range(num_samples):
    sample = dataset[i]
    mfccs = sample['mfccs'].unsqueeze(0).to(device)
    spectral_centroid = sample['spectral_centroid'].unsqueeze(0).to(device)
    rms = sample['rms'].unsqueeze(0).to(device)
    audio_features = torch.cat([mfccs, spectral_centroid, rms], dim=1)

    with torch.no_grad():
        recon_audio, _, _ = model(audio_features)
        mse = torch.mean((audio_features - recon_audio) ** 2).item()
        print(f"Sample {i} ({dataset.audio_files[i]}): MSE = {mse}")
        total_mse += mse

avg_mse = total_mse / num_samples
print(f"Average Mean Squared Error over {num_samples} samples: {avg_mse}")
