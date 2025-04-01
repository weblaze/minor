# D:\musicc\minor\models\autoencoder\mapping_network.py
import torch
import torch.nn as nn

class MappingNetwork(nn.Module):
    def __init__(self, audio_latent_dim=256, image_latent_dim=256):
        super(MappingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(audio_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, image_latent_dim * 2),  # mu + logvar
        )

    def forward(self, x):
        output = self.net(x)
        mu, logvar = output.chunk(2, dim=-1)
        # Check for NaN/Inf in mu and logvar
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("Warning: MappingNetwork mu contains NaN or Inf")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print("Warning: MappingNetwork logvar contains NaN or Inf")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Check for NaN/Inf in z
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: MappingNetwork z contains NaN or Inf after reparameterization")
        return z

class InverseMappingNetwork(nn.Module):
    def __init__(self, image_latent_dim=256, audio_latent_dim=256):
        super(InverseMappingNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(image_latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, audio_latent_dim * 2),  # mu + logvar
        )

    def forward(self, x):
        output = self.net(x)
        mu, logvar = output.chunk(2, dim=-1)
        # Check for NaN/Inf in mu and logvar
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("Warning: InverseMappingNetwork mu contains NaN or Inf")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print("Warning: InverseMappingNetwork logvar contains NaN or Inf")
        return mu, logvar

    def reparameterize(self, mu, logvar):
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Check for NaN/Inf in z
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: InverseMappingNetwork z contains NaN or Inf after reparameterization")
        return z