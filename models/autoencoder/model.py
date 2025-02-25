import torch
import torch.nn as nn
import torch.nn.functional as F

class ResBlock(nn.Module):
    def __init__(self, in_channels):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.in1 = nn.InstanceNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1)
        self.in2 = nn.InstanceNorm2d(in_channels)
        
    def forward(self, x):
        residual = x
        out = F.relu(self.in1(self.conv1(x)))
        out = self.in2(self.conv2(out))
        out += residual
        out = F.relu(out)
        return out

class AttentionBlock(nn.Module):
    def __init__(self, in_channels):
        super(AttentionBlock, self).__init__()
        self.query = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.key = nn.Conv2d(in_channels, in_channels//8, kernel_size=1)
        self.value = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
        
    def forward(self, x):
        batch_size, channels, height, width = x.size()
        
        q = self.query(x).view(batch_size, -1, height*width).permute(0, 2, 1)
        k = self.key(x).view(batch_size, -1, height*width)
        v = self.value(x).view(batch_size, -1, height*width)
        
        attention = F.softmax(torch.bmm(q, k), dim=-1)
        out = torch.bmm(v, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)
        
        return self.gamma * out + x

class AudioToImageAutoencoder(nn.Module):
    def __init__(self, audio_dim=191, latent_dim=1024, image_size=128):
        super(AudioToImageAutoencoder, self).__init__()
        
        # Store dimensions
        self.audio_dim = audio_dim
        self.latent_dim = latent_dim
        self.image_size = image_size
        
        # Audio feature processing with wider layers
        self.audio_processor = nn.Sequential(
            nn.Linear(audio_dim, 1024),
            nn.LayerNorm(1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(1024, 2048),
            nn.LayerNorm(2048),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1),
            
            nn.Linear(2048, 4096),
            nn.LayerNorm(4096),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # VAE encoder (mu and logvar)
        self.encoder_mu = nn.Linear(4096, latent_dim)
        self.encoder_logvar = nn.Linear(4096, latent_dim)
        
        # Initial projection to 8x8 spatial dimension
        initial_size = image_size // 16  # Start from 8x8 for 128x128 output
        self.projection = nn.Sequential(
            nn.Linear(latent_dim, initial_size * initial_size * 512),
            nn.LayerNorm(initial_size * initial_size * 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.1)
        )
        
        # Decoder blocks with progressive upsampling
        self.decoder_blocks = nn.ModuleList([
            # 8x8 -> 16x16
            nn.Sequential(
                nn.ConvTranspose2d(512, 512, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(512),
                nn.LeakyReLU(0.2),
                ResBlock(512),
                AttentionBlock(512)
            ),
            # 16x16 -> 32x32
            nn.Sequential(
                nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(256),
                nn.LeakyReLU(0.2),
                ResBlock(256),
                AttentionBlock(256)
            ),
            # 32x32 -> 64x64
            nn.Sequential(
                nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(128),
                nn.LeakyReLU(0.2),
                ResBlock(128),
                AttentionBlock(128)
            ),
            # 64x64 -> 128x128
            nn.Sequential(
                nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
                nn.InstanceNorm2d(64),
                nn.LeakyReLU(0.2),
                ResBlock(64),
                AttentionBlock(64)
            )
        ])
        
        # Style blocks for artistic enhancement
        self.style_blocks = nn.ModuleList([
            ResBlock(64),
            ResBlock(64),
            ResBlock(64)
        ])
        
        # Color refinement with improved architecture
        self.color_refinement = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.InstanceNorm2d(32),
            nn.LeakyReLU(0.2),
            ResBlock(32),
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.InstanceNorm2d(16),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 3, kernel_size=1),
            nn.Hardtanh()  # Use Hardtanh instead of Sigmoid for better gradient flow
        )
        
        # Initialize weights with improved scaling
        self.apply(self._init_weights)
        
        # Flag for gradient checkpointing
        self.use_checkpointing = False
    
    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d, nn.Linear)):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='leaky_relu', a=0.2)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
    
    def enable_gradient_checkpointing(self):
        """Enable gradient checkpointing for memory efficiency."""
        self.use_checkpointing = True
        
        # Enable gradient checkpointing for all modules
        for module in self.modules():
            if isinstance(module, (nn.Sequential, ResBlock, AttentionBlock)):
                for param in module.parameters():
                    param.requires_grad = True
    
    def encode(self, audio_features):
        """Encode audio features to latent space."""
        # Process audio features with checkpointing
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.audio_processor,
                audio_features,
                use_reentrant=False
            )
        else:
            x = self.audio_processor(audio_features)
        
        # Get mu and logvar
        mu = self.encoder_mu(x)
        logvar = self.encoder_logvar(x)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick with improved numerical stability."""
        if self.training:
            std = torch.exp(0.5 * torch.clamp(logvar, -10, 10))
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu
    
    def decode(self, z):
        """Decode latent vector to image."""
        # Project to spatial features
        batch_size = z.size(0)
        initial_size = self.image_size // 16
        
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.projection,
                z,
                use_reentrant=False
            )
        else:
            x = self.projection(z)
        
        x = x.view(batch_size, 512, initial_size, initial_size)
        
        # Apply decoder blocks
        for block in self.decoder_blocks:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False
                )
            else:
                x = block(x)
        
        # Apply style processing
        for block in self.style_blocks:
            if self.use_checkpointing and self.training:
                x = torch.utils.checkpoint.checkpoint(
                    block,
                    x,
                    use_reentrant=False
                )
            else:
                x = block(x)
        
        # Final refinement
        if self.use_checkpointing and self.training:
            x = torch.utils.checkpoint.checkpoint(
                self.color_refinement,
                x,
                use_reentrant=False
            )
        else:
            x = self.color_refinement(x)
        
        # Scale to [0, 1] range
        return x * 0.5 + 0.5
    
    def forward(self, audio_features):
        # Encode
        mu, logvar = self.encode(audio_features)
        
        # Reparameterize
        z = self.reparameterize(mu, logvar)
        
        # Decode
        output = self.decode(z)
        
        return output, mu, logvar
