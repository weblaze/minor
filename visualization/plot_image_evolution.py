import matplotlib.pyplot as plt
import glob
import os
import imageio

# Directory setup
SAVE_DIR = r"D:\musicc\minor\visualization"
OUTPUT_DIR = r"D:\musicc\minor\datasets\reconstructed_images"  # Assumed output dir from train_image_vae.py
os.makedirs(SAVE_DIR, exist_ok=True)

# Load generated images (replace with actual path if different)
image_files = sorted(glob.glob(os.path.join(OUTPUT_DIR, 'recon_epoch_*.png')))
images = [imageio.imread(f) for f in image_files[:5]]  # Limit to 5 epochs for display

# Plotting
fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
for ax, img, file in zip(axes, images, image_files):
    ax.imshow(img)
    ax.axis('off')
    ax.set_title(os.path.basename(file))
plt.suptitle('Generated Image Evolution')
plt.tight_layout()
plt.savefig(os.path.join(SAVE_DIR, 'image_evolution.png'))
plt.close()

# Simulated FID scores (replace with actual computation)
epochs = [10, 20, 30, 40, 50]
fid_scores = [50.0, 40.0, 30.0, 20.0, 15.0]  # Simulated FID decrease
plt.figure(figsize=(10, 6))
plt.plot(epochs, fid_scores, label='FID', marker='o')
plt.xlabel('Epoch')
plt.ylabel('FID Score')
plt.title('Generated Image Quality (FID)')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(SAVE_DIR, 'fid_scores.png'))
plt.close()

print("Image evolution and FID scores saved in", SAVE_DIR)