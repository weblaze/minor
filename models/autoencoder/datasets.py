# D:\musicc\minor\models\autoencoder\datasets.py
import os
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageOps
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, image_path, train=True):
        self.image_path = image_path
        self.image_files = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.train = train
        # Define transforms without rotation
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Resize to 128x128
            transforms.ToTensor(),  # Convert to tensor
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize to [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        # Fix EXIF orientation
        image = ImageOps.exif_transpose(image)  # Correctly handles EXIF orientation
        image = self.transform(image)
        return image

class AudioFeatureDataset(Dataset):
    def __init__(self, audio_path, normalize=True, train=True):
        self.audio_path = audio_path
        self.audio_files = [f for f in os.listdir(audio_path) if f.endswith('.npy')]  # Changed to .npy
        self.normalize = normalize
        self.train = train
        # Debug: Print the number of files found
        print(f"Found {len(self.audio_files)} audio feature files in {audio_path}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_path, self.audio_files[idx])
        audio_features = np.load(audio_path)  # Load .npy file
        audio_features = torch.tensor(audio_features, dtype=torch.float32)  # Convert to tensor
        if self.normalize:
            audio_features = (audio_features - audio_features.mean()) / (audio_features.std() + 1e-6)
        return audio_features