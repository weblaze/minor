import torch
import torch.nn as nn
import torch.nn.functional as F

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
            
            # Split into mean and variance for variational component
            nn.Linear(2048, latent_dim * 2)
        )
        
        # Reshape layer to prepare for deconvolution
        self.latent_to_image = nn.Sequential(
            nn.Linear(latent_dim, 8 * 8 * 512),  # Increased initial size
            nn.BatchNorm1d(8 * 8 * 512),
            nn.LeakyReLU(0.2)
        )
        
        # Decoder with residual connections
        self.decoder_block1 = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block2 = nn.Sequential(
            nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2)
        )
        
        self.decoder_block3 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        # Style modulation layers
        self.style_mod1 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2)
        )
        
        self.style_mod2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.InstanceNorm2d(64),
            nn.LeakyReLU(0.2)
        )
        
        # Color enhancement layers
        self.color_enhance = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(32, 3, kernel_size=1),
            nn.Tanh()
        )
        
        # Texture enhancement layers
        self.texture_enhance = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            
            nn.Conv2d(16, 3, kernel_size=3, padding=1),
            nn.Tanh()
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, audio_features):
        # Encode audio to latent space with variational component
        x = self.encoder(audio_features)
        mu, logvar = torch.split(x, self.latent_dim, dim=1)
        z = self.reparameterize(mu, logvar)
        
        # Reshape for deconvolution
        x = self.latent_to_image(z)
        x = x.view(-1, 512, 8, 8)
        
        # Decoder with residual connections
        x1 = self.decoder_block1(x)
        x2 = self.decoder_block2(x1)
        x3 = self.decoder_block3(x2)
        
        # Apply style modulation
        style = self.style_mod1(x3)
        style = self.style_mod2(style + x3)  # Residual connection
        
        # Enhance colors
        color = self.color_enhance(style)
        
        # Enhance texture
        texture = self.texture_enhance(color)
        
        # Combine color and texture with residual connection
        output = (color + texture) * 0.5
        
        # Ensure output is in [0, 1] range with enhanced contrast
        output = ((output + 1) * 0.5).clamp(0, 1)
        
        # Apply final contrast enhancement
        output = (output - output.mean()) * 1.2 + 0.5
        output = output.clamp(0, 1)
        
        return output, mu, logvar
