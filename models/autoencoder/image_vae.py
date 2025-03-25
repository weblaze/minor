import torch
import torch.nn as nn

class ImageVAE(nn.Module):
    def __init__(self, latent_dim=64):
        super(ImageVAE, self).__init__()
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
            ),
            nn.Sequential(
                nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
            ),
            nn.Sequential(
                nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(256),
            ),
        ])
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)  # 128x128 → 8x8 after 4 stride-2 layers
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, 256 * 8 * 8)
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(128),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(64),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm2d(32),
            ),
            nn.Sequential(
                nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
                nn.Sigmoid(),
            ),
        ])

    def encode(self, x):
        skips = []
        h = x
        for layer in self.encoder_layers:
            h = layer(h)
            skips.append(h)
        h = h.view(h.size(0), -1)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar, skips

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, skips):
        h = self.decoder_input(z).view(-1, 256, 8, 8)
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
        return h

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar