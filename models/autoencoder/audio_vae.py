# D:\musicc\minor\models\autoencoder\audio_vae.py
import torch
import torch.nn as nn

class AudioVAE(nn.Module):
    def __init__(self, input_channels=20, time_steps=216, latent_dim=256):
        super(AudioVAE, self).__init__()
        self.time_steps = time_steps
        self.encoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(input_channels, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            ),
            nn.Sequential(
                nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
            ),
            nn.Sequential(
                nn.Conv1d(64, 128, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(128),
            ),
        ])
        self.encoder_final_time_steps = time_steps
        for _ in range(3):  # 3 stride-2 layers
            self.encoder_final_time_steps = (self.encoder_final_time_steps + 1) // 2
        self.conv_output_size = 128 * self.encoder_final_time_steps
        self.fc_mu = nn.Linear(self.conv_output_size, latent_dim)
        self.fc_logvar = nn.Linear(self.conv_output_size, latent_dim)

        self.decoder_input = nn.Linear(latent_dim, self.conv_output_size)
        self.decoder_layers = nn.ModuleList([
            nn.Sequential(
                nn.ConvTranspose1d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(64),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.ReLU(),
                nn.BatchNorm1d(32),
            ),
            nn.Sequential(
                nn.ConvTranspose1d(32, input_channels, kernel_size=3, stride=2, padding=1, output_padding=1),
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
        # Check for NaN/Inf in mu and logvar
        if torch.isnan(mu).any() or torch.isinf(mu).any():
            print("Warning: mu contains NaN or Inf")
        if torch.isnan(logvar).any() or torch.isinf(logvar).any():
            print("Warning: logvar contains NaN or Inf")
        return mu, logvar, skips

    def reparameterize(self, mu, logvar):
        # Clamp logvar to prevent numerical instability
        logvar = torch.clamp(logvar, min=-10, max=10)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + eps * std
        # Check for NaN/Inf in z
        if torch.isnan(z).any() or torch.isinf(z).any():
            print("Warning: z contains NaN or Inf after reparameterization")
        return z

    def decode(self, z, skips):
        h = self.decoder_input(z).view(-1, 128, self.encoder_final_time_steps)
        for i, layer in enumerate(self.decoder_layers):
            h = layer(h)
        return h

    def forward(self, x):
        mu, logvar, skips = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, skips), mu, logvar