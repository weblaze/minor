import matplotlib.pyplot as plt
import numpy as np
import os

# Directory setup
SAVE_DIR = r"D:\musicc\minor\visualization"
os.makedirs(SAVE_DIR, exist_ok=True)

# Simulated cycle consistency losses (replace with actual logs)
epochs = list(range(1, 51))
cycle_audio_losses = [0.002 * (50 / e) if e > 0 else 0.2 for e in epochs]
cycle_image_losses = [0.001 * (50 / e) if e > 0 else 0.1 for e in epochs]

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, cycle_audio_losses, label='Cycle Audio Loss')
plt.plot(epochs, cycle_image_losses, label='Cycle Image Loss')
plt.xlabel('Epoch')
plt.ylabel('Cycle Consistency Loss')
plt.title('MappingNetwork Cycle Consistency')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'cycle_consistency.png'))
plt.close()

print("Cycle consistency errors saved in", SAVE_DIR)