import numpy as np
from PIL import Image

def analyze_image(path):
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img) / 255.0  # Normalize to [0, 1]
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        pmin = arr.min(axis=(0, 1))
        pmax = arr.max(axis=(0, 1))
        print(f"File: {path}")
        print(f"  Mean RGB: {mean}")
        print(f"  Std RGB:  {std}")
        print(f"  Min RGB:  {pmin}")
        print(f"  Max RGB:  {pmax}")
        print("-" * 50)
    except Exception as e:
        print(f"Error reading {path}: {e}")

if __name__ == "__main__":
    analyze_image("outputs/01_conditioned_autoencoder/genre_gallery.png")
    analyze_image("generated_images/_archive/aanewala.png")
    analyze_image("generated_images/_archive/brakence.png")
    analyze_image("generated_images/_archive/edm1.png")
    analyze_image("evaluation/results/vae_reconstruction_test.png")
