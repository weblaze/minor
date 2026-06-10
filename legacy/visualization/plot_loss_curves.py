import matplotlib.pyplot as plt
import numpy as np
import os

# Directory setup
SAVE_DIR = r"D:\musicc\minor\visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

# Simulated training logs (replace with actual logs from train_audio_vae.py, train_image_vae.py, train_mapping_network.py)
epochs = list(range(1, 101))
audio_losses = [0.05 * (100 / e) if e > 0 else 5.0 for e in epochs]  # Simulated total loss
audio_mse = [0.03 * (100 / e) if e > 0 else 3.0 for e in epochs]    # Simulated MSE
audio_kl = [0.02 * (100 / e) if e > 0 else 2.0 for e in epochs]     # Simulated KL
image_losses = [0.04 * (100 / e) if e > 0 else 4.0 for e in epochs]
image_mse = [0.02 * (100 / e) if e > 0 else 2.0 for e in epochs]
image_style = [0.01 * (100 / e) if e > 0 else 1.0 for e in epochs]
image_kl = [0.01 * (100 / e) if e > 0 else 1.0 for e in epochs]
map_losses = [0.03 * (100 / e) if e > 0 else 3.0 for e in epochs]
map_kl = [0.005 * (100 / e) if e > 0 else 0.5 for e in epochs]
map_cycle_audio = [0.002 * (100 / e) if e > 0 else 0.2 for e in epochs]
map_div = [0.001 * (100 / e) if e > 0 else 0.1 for e in epochs]
map_mmd = [0.003 * (100 / e) if e > 0 else 0.3 for e in epochs]

# Plotting AudioVAE Losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, audio_losses, label='Total Loss')
plt.plot(epochs, audio_mse, label='Reconstruction Loss')
plt.plot(epochs, audio_kl, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('AudioVAE Training Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'audio_vae_loss_curves.png'))
plt.close()

# Plotting ImageVAE Losses
plt.figure(figsize=(10, 6))
plt.plot(epochs, image_losses, label='Total Loss')
plt.plot(epochs, image_mse, label='MSE')
plt.plot(epochs, image_style, label='Style Loss')
plt.plot(epochs, image_kl, label='KL Divergence')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('ImageVAE Training Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'image_vae_loss_curves.png'))
plt.close()

# Plotting MappingNetwork Losses
plt.figure(figsize=(12, 6))
plt.plot(epochs, map_losses, label='Mapping Loss')
plt.plot(epochs, map_kl, label='KL Loss')
plt.plot(epochs, map_cycle_audio, label='Cycle Audio Loss')
plt.plot(epochs, map_div, label='Diversity Loss')
plt.plot(epochs, map_mmd, label='MMD Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('MappingNetwork Training Loss Curves')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'mapping_network_loss_curves.png'))
plt.close()

print("Loss curves saved in", SAVE_DIR)