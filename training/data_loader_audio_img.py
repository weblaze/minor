import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from PIL import Image
import os
from pathlib import Path
from torchvision import transforms
import logging
from typing import Tuple, List, Optional
import random

logger = logging.getLogger(__name__)

class AudioImageDataset(Dataset):
    def __init__(self, audio_features_path: str, image_folder: str,
                 normalize_features: bool = True, augment_images: bool = True,
                 augment_audio: bool = True):
        """
        Dataset for audio features and corresponding images.
        
        Args:
            audio_features_path: Path to numpy file containing audio features
            image_folder: Path to folder containing images
            normalize_features: Whether to normalize audio features
            augment_images: Whether to apply data augmentation to images
            augment_audio: Whether to apply data augmentation to audio features
        """
        self.audio_features_path = Path(audio_features_path)
        self.image_folder = Path(image_folder)
        self.augment_audio = augment_audio
        
        # Load and process audio features
        self.audio_features = self._load_audio_features(normalize_features)
        
        # Setup image transforms
        self.transform = self._get_image_transforms(augment_images)
        
        # Get and validate image files
        self.image_files = self._get_image_files()
        
        # Setup audio augmentation parameters
        self.audio_aug_params = {
            'pitch_shift': (-2, 2),  # Semitones
            'time_stretch': (0.9, 1.1),  # Rate
            'noise_level': (0.0, 0.05),  # Amplitude
            'freq_mask': (0.0, 0.2),  # Percentage of frequencies to mask
            'time_mask': (0.0, 0.2)  # Percentage of time steps to mask
        }
        
        logger.info(f"Dataset initialized with {len(self)} pairs")
        logger.info(f"Audio feature dimension: {self.audio_features.shape[1]}")
        logger.info(f"Number of unique images: {len(set(self.image_files))}")
    
    def _load_audio_features(self, normalize: bool) -> np.ndarray:
        """Load and optionally normalize audio features."""
        try:
            features = np.load(self.audio_features_path)
            
            if normalize:
                # Normalize each feature dimension independently
                mean = np.mean(features, axis=0, keepdims=True)
                std = np.std(features, axis=0, keepdims=True)
                std[std == 0] = 1  # Prevent division by zero
                features = (features - mean) / std
                
                logger.info("Audio features normalized")
            
            return features
            
        except Exception as e:
            logger.error(f"Error loading audio features: {str(e)}")
            raise
    
    def _get_image_transforms(self, augment: bool) -> transforms.Compose:
        """Get image transformation pipeline."""
        transform_list = [
            transforms.Resize((64, 64)),
            transforms.ToTensor(),
        ]
        
        if augment:
            transform_list.extend([
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomAffine(
                    degrees=15, translate=(0.1, 0.1),
                    scale=(0.9, 1.1), shear=5
                ),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.1
                )
            ])
        
        transform_list.append(transforms.Normalize(mean=[0.5, 0.5, 0.5],
                                                std=[0.5, 0.5, 0.5]))
        
        return transforms.Compose(transform_list)
    
    def _get_image_files(self) -> List[Path]:
        """Get and validate image files."""
        image_files = list(self.image_folder.glob('*.jpg')) + \
                     list(self.image_folder.glob('*.png'))
        
        if not image_files:
            raise ValueError(f"No images found in {self.image_folder}")
        
        # If we have more audio features than images, repeat the images
        if len(image_files) < len(self.audio_features):
            repeat_times = len(self.audio_features) // len(image_files) + 1
            image_files = image_files * repeat_times
            image_files = image_files[:len(self.audio_features)]
            
            logger.warning(f"Repeated {len(set(image_files))} images "
                         f"{repeat_times} times to match {len(self.audio_features)} "
                         "audio features")
        
        return image_files
    
    def __len__(self) -> int:
        return len(self.audio_features)
    
    def _augment_audio_features(self, features: torch.Tensor) -> torch.Tensor:
        """Apply augmentation to audio features."""
        if not self.augment_audio:
            return features
        
        # Convert to numpy for easier manipulation
        features = features.numpy()
        
        # Random pitch shift simulation
        if random.random() < 0.5:
            shift = random.uniform(*self.audio_aug_params['pitch_shift'])
            features = self._simulate_pitch_shift(features, shift)
        
        # Random time stretch simulation
        if random.random() < 0.5:
            rate = random.uniform(*self.audio_aug_params['time_stretch'])
            features = self._simulate_time_stretch(features, rate)
        
        # Add random noise
        if random.random() < 0.3:
            noise_amp = random.uniform(*self.audio_aug_params['noise_level'])
            features += np.random.normal(0, noise_amp, features.shape)
        
        # Frequency masking
        if random.random() < 0.3:
            mask_size = int(random.uniform(*self.audio_aug_params['freq_mask']) * features.shape[0])
            start = random.randint(0, features.shape[0] - mask_size)
            features[start:start + mask_size] = 0
        
        # Time masking
        if random.random() < 0.3:
            mask_size = int(random.uniform(*self.audio_aug_params['time_mask']) * features.shape[0])
            start = random.randint(0, features.shape[0] - mask_size)
            features[start:start + mask_size] = 0
        
        return torch.FloatTensor(features)
    
    def _simulate_pitch_shift(self, features: np.ndarray, shift: float) -> np.ndarray:
        """Simulate pitch shift effect on features."""
        # Adjust frequency-related features
        freq_indices = [0, 1, 2]  # Indices of frequency-related features
        shift_factor = 2 ** (shift / 12)  # Convert semitones to frequency ratio
        
        features = features.copy()
        features[freq_indices] *= shift_factor
        return features
    
    def _simulate_time_stretch(self, features: np.ndarray, rate: float) -> np.ndarray:
        """Simulate time stretch effect on features."""
        # Adjust temporal features
        tempo_idx = 3  # Index of tempo feature
        features = features.copy()
        features[tempo_idx] *= rate
        return features
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        try:
            # Get audio features
            audio_feature = torch.FloatTensor(self.audio_features[idx])
            
            # Apply audio augmentation
            if self.augment_audio:
                audio_feature = self._augment_audio_features(audio_feature)
            
            # Load and transform image
            image_path = self.image_files[idx]
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                logger.error(f"Error loading image {image_path}: {str(e)}")
                # Use a random different image as fallback
                fallback_idx = (idx + 1) % len(self)
                image = Image.open(self.image_files[fallback_idx]).convert('RGB')
            
            image = self.transform(image)
            
            return audio_feature, image
            
        except Exception as e:
            logger.error(f"Error getting item {idx}: {str(e)}")
            raise

def get_dataloader(audio_features_path: str, image_folder: str, 
                  batch_size: int = 32, shuffle: bool = True,
                  num_workers: int = 4, normalize_features: bool = True,
                  augment_images: bool = True, pin_memory: bool = True,
                  prefetch_factor: int = 2) -> DataLoader:
    """
    Create a DataLoader for the audio-image dataset.
    
    Args:
        audio_features_path: Path to numpy file containing audio features
        image_folder: Path to folder containing images
        batch_size: Number of samples per batch
        shuffle: Whether to shuffle the data
        num_workers: Number of worker processes for data loading
        normalize_features: Whether to normalize audio features
        augment_images: Whether to apply data augmentation to images
        pin_memory: Whether to pin memory for faster GPU transfer
        prefetch_factor: Number of batches to prefetch per worker
    
    Returns:
        DataLoader instance
    """
    try:
        dataset = AudioImageDataset(
            audio_features_path=audio_features_path,
            image_folder=image_folder,
            normalize_features=normalize_features,
            augment_images=augment_images
        )
        
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            prefetch_factor=prefetch_factor if num_workers > 0 else None
        )
        
    except Exception as e:
        logger.error(f"Error creating DataLoader: {str(e)}")
        raise 