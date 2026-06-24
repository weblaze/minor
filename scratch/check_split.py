import numpy as np
from PIL import Image

def analyze_split(path):
    img = Image.open(path).convert("RGB")
    arr = np.array(img) / 255.0
    h, w, c = arr.shape
    
    # Split top half (originals) and bottom half (reconstructions)
    top = arr[:h//2, :, :]
    bottom = arr[h//2:, :, :]
    
    print(f"File: {path}")
    print(f"Top Half (Originals):")
    print(f"  Mean RGB: {top.mean(axis=(0, 1))}")
    print(f"  Std RGB:  {top.std(axis=(0, 1))}")
    print(f"Bottom Half (Reconstructions):")
    print(f"  Mean RGB: {bottom.mean(axis=(0, 1))}")
    print(f"  Std RGB:  {bottom.std(axis=(0, 1))}")
    print("-" * 50)

if __name__ == "__main__":
    analyze_split("evaluation/results/vae_reconstruction_test.png")
