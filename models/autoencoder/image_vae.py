import torch
import torch.nn as nn

class ImageVAE(nn.Module):
    def __init__(self, latent_channels=8):  
        super(ImageVAE, self).__init__()
        self.latent_channels = latent_channels
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1), # 16x16
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128),
        )
        
        # Latent projections (Spatial)
        self.fc_mu = nn.Conv2d(128, latent_channels, kernel_size=1)
        self.fc_logvar = nn.Conv2d(128, latent_channels, kernel_size=1)

        # Decoder: 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder_input = nn.Conv2d(latent_channels, 128, kernel_size=1)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),   # 64x64
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),    # 128x128
            nn.Tanh() # Output in [-1, 1]
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, []

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips=None):
        h = self.decoder_input(z)
        h = self.decoder(h)
        return h

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar