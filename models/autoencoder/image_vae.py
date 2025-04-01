# D:\musicc\minor\models\autoencoder\image_vae.py
import torch
import torch.nn as nn

class ImageVAE(nn.Module):
    def __init__(self, latent_dim=256):
        super(ImageVAE, self).__init__()
        self.latent_dim = latent_dim
        
        # Encoder: 128x128 -> 64x64 -> 32x32 -> 16x16 -> 8x8 -> 4x4
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1),  # 128x128 -> 64x64
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  # 64x64 -> 32x32
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(128),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  # 32x32 -> 16x16
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(256),
            ),
            nn.Sequential(
                nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1),  # 16x16 -> 8x8
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(512),
            ),
            nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=4, stride=2, padding=1),  # 8x8 -> 4x4
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(512),
            ),
        ])
        self.fc_mu = nn.Linear(512 * 4 * 4, latent_dim)  # 8192 -> 256
        self.fc_logvar = nn.Linear(512 * 4 * 4, latent_dim)

        # Decoder: 4x4 -> 8x8 -> 16x16 -> 32x32 -> 64x64 -> 128x128
        self.decoder_input = nn.Linear(latent_dim, 512 * 4 * 4)
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),  # 4x4 -> 8x8
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(512),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),  # 8x8 -> 16x16
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(256),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  # 16x16 -> 32x32
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(128),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 32x32 -> 64x64
                nn.ReLU(inplace=False),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1),  # 64x64 -> 128x128
                nn.Tanh(),  # Output in [-1, 1] to match dataset normalization
            ),
        ])

    def encode(self, x):
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
        h = h.view(h.size(0), -1)  # [batch_size, 8192]
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips=None):
        h = self.decoder_input(z).view(-1, 512, 4, 4)
        for layer in self.decoder_layers:
            h = layer(h)
        return h

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar