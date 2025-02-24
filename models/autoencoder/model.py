import torch
import torch.nn as nn

class AudioToImageAutoencoder(nn.Module):
    def __init__(self, audio_dim=191, latent_dim=512):
        super(AudioToImageAutoencoder, self).__init__()
        
        # Encoder - Transform audio features to latent space
        self.encoder = nn.Sequential(
            nn.Linear(audio_dim, 1024),
            nn.BatchNorm1d(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(1024, 2048),
            nn.BatchNorm1d(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            
            nn.Linear(2048, latent_dim),
            nn.BatchNorm1d(latent_dim),
            nn.Tanh()
        )
        
        # Reshape layer to prepare for deconvolution
        self.latent_to_image = nn.Sequential(
            nn.Linear(latent_dim, 4 * 4 * 512),
            nn.BatchNorm1d(4 * 4 * 512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder - Generate abstract art
        self.decoder = nn.Sequential(
            # 4x4 -> 8x8
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2),
            
            # 8x8 -> 16x16
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2),
            
            # 16x16 -> 32x32
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2),
            
            # 32x32 -> 64x64
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2),
            
            # Final layers for artistic effects
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )
        
        # Style enhancement layers
        self.style_enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def forward(self, audio_features):
        # Encode audio to latent space
        latent = self.encoder(audio_features)
        
        # Reshape for deconvolution
        x = self.latent_to_image(latent)
        x = x.view(-1, 512, 4, 4)
        
        # Generate base image
        x = self.decoder(x)
        
        # Enhance artistic style
        x = self.style_enhance(x)
        
        # Ensure output is in [0, 1] range
        return (x + 1) / 2
