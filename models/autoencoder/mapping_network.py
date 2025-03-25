import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, audio_latent_dim=64, image_latent_dim=64):
        super(MappingNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(audio_latent_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, image_latent_dim * 2),
        )

    def forward(self, z_audio):
        output = self.network(z_audio)
        mu, logvar = output.chunk(2, dim=-1)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std