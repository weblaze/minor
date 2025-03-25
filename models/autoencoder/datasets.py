import torch
from torch.utils.data import Dataset
import numpy as np
import os
from PIL import Image
import torchvision.transforms as transforms

class AudioFeatureDataset(Dataset):
    def __init__(self, feature_dir, normalize=True, train=True):
        self.feature_dir = feature_dir
        self.files = [f for f in os.listdir(feature_dir) if f.endswith(".npy")]
        self.normalize = normalize
        self.train = train

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        feature_path = os.path.join(self.feature_dir, self.files[idx])
        features = np.load(feature_path)
        features = torch.FloatTensor(features)  # Shape: (time_steps, 20)
        if self.train:
            # Random scaling
            scale = torch.FloatTensor(1).uniform_(0.7, 1.3)
            features = features * scale
            # Random noise
            noise = torch.randn_like(features) * 0.05
            features = features + noise
            # Random time shift
            shift = int(torch.randint(-10, 10, (1,)).item())
            features = torch.roll(features, shifts=shift, dims=0)
        if self.normalize:
            features = (features - features.min()) / (features.max() - features.min() + 1e-8)
        return features

class ImageDataset(Dataset):
    def __init__(self, image_dir, train=True):
        self.image_dir = image_dir
        self.files = [f for f in os.listdir(image_dir) if f.endswith((".png", ".jpg", ".jpeg"))]
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),  # Train at 128x128
            transforms.RandomHorizontalFlip() if train else transforms.Lambda(lambda x: x),
            transforms.RandomRotation(20) if train else transforms.Lambda(lambda x: x),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1) if train else transforms.Lambda(lambda x: x),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)) if train else transforms.Lambda(lambda x: x),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        image_path = os.path.join(self.image_dir, self.files[idx])
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image