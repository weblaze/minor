import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms

class AudioImageDataset(Dataset):
    def __init__(self, audio_features_path, image_folder):
        # Load audio features
        self.audio_features = np.load(audio_features_path)
        
        # Setup image transforms
        self.transform = transforms.Compose([
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ])
        
        # Get image files
        self.image_folder = Path(image_folder)
        self.image_files = list(self.image_folder.glob('*.jpg')) + list(self.image_folder.glob('*.png'))
        
        # If we have more audio features than images, repeat the images
        if len(self.image_files) < len(self.audio_features):
            repeat_times = len(self.audio_features) // len(self.image_files) + 1
            self.image_files = self.image_files * repeat_times
            self.image_files = self.image_files[:len(self.audio_features)]
        
        print(f"Dataset size: {len(self.audio_features)} pairs")
        print(f"Audio feature dimension: {self.audio_features.shape[1]}")
        
    def __len__(self):
        return len(self.audio_features)
    
    def __getitem__(self, idx):
        # Get audio features
        audio_feature = torch.FloatTensor(self.audio_features[idx])
        
        # Load and transform image
        image = Image.open(self.image_files[idx]).convert('RGB')
        image = self.transform(image)
        
        return audio_feature, image

def get_dataloader(audio_features_path, image_folder, batch_size=32, shuffle=True):
    dataset = AudioImageDataset(audio_features_path, image_folder)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle) 