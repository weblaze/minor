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
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = self.transform(image)
        return image

class AudioFeatureDataset(Dataset):
    def __init__(self, audio_path, normalize=True, train=True):
        self.audio_path = audio_path
        self.audio_files = [f for f in os.listdir(audio_path) if f.endswith('.npy')]
        self.normalize = normalize
        self.train = train
        print(f"Found {len(self.audio_files)} audio feature files in {audio_path}")

    def __len__(self):
        return len(self.audio_files)

    def __getitem__(self, idx):
        audio_path = os.path.join(self.audio_path, self.audio_files[idx])
        try:
            loaded_data = np.load(audio_path, allow_pickle=True)
            if isinstance(loaded_data, np.ndarray) and loaded_data.size == 1 and isinstance(loaded_data.item(), dict):
                features = loaded_data.item()  # New format: dictionary
            elif isinstance(loaded_data, np.ndarray) and loaded_data.ndim == 2:  # Old format: raw MFCCs
                print(f"Warning: Converting old format file {audio_path} to dictionary")
                features = {
                    'mfccs': loaded_data,  # Assume shape (time_steps, N_MFCC)
                    'spectral_centroid': np.zeros((loaded_data.shape[0], 1)),  # Placeholder
                    'rms': np.zeros((loaded_data.shape[0], 1)),  # Placeholder
                    'tempo': np.array([0.0])  # Placeholder, shape (1,)
                }
            else:
                raise ValueError(f"Unsupported data format in {audio_path}: Type {type(loaded_data)}, Shape {loaded_data.shape if isinstance(loaded_data, np.ndarray) else 'N/A'}")

            mfccs = features['mfccs']
            spectral_centroid = features['spectral_centroid']
            rms = features['rms']
            tempo = features['tempo']

            # Convert to tensors and ensure consistent shapes
            mfccs = torch.tensor(mfccs, dtype=torch.float32)  # Shape: (time_steps, N_MFCC)
            spectral_centroid = torch.tensor(spectral_centroid, dtype=torch.float32)  # Shape: (time_steps, 1)
            rms = torch.tensor(rms, dtype=torch.float32)  # Shape: (time_steps, 1)
            tempo = torch.tensor(tempo, dtype=torch.float32)  # Shape: (1,) or scalar
            if tempo.dim() == 0:
                tempo = tempo.unsqueeze(0)  # Ensure shape (1,)
            elif tempo.dim() == 2 and tempo.size(1) == 1:
                tempo = tempo.squeeze(1)  # Ensure shape (1,)

            # Transpose to (channels, time_steps) for concatenation
            mfccs = mfccs.transpose(0, 1)  # Shape: (N_MFCC, time_steps)
            spectral_centroid = spectral_centroid.transpose(0, 1)  # Shape: (1, time_steps)
            rms = rms.transpose(0, 1)  # Shape: (1, time_steps)

            # Debugging: Print shapes
            print(f"Debug - {audio_path}: mfccs shape {mfccs.shape}, spectral_centroid shape {spectral_centroid.shape}, rms shape {rms.shape}, tempo shape {tempo.shape}")

            if self.normalize:
                mfccs = (mfccs - mfccs.mean()) / (mfccs.std() + 1e-6)
                spectral_centroid = (spectral_centroid - spectral_centroid.mean()) / (spectral_centroid.std() + 1e-6)
                rms = (rms - rms.mean()) / (rms.std() + 1e-6)

            audio_features = {
                'mfccs': mfccs,
                'spectral_centroid': spectral_centroid,
                'rms': rms,
                'tempo': tempo
            }
            return audio_features

        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            raise