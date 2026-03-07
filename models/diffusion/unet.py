import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings

class Block(nn.Module):
    def __init__(self, in_ch, out_ch, time_emb_dim, up=False, has_skip=True):
        super().__init__()
        self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
        if up:
            # If it's an up-block with a skip connection, it received cat(upsampled, skip)
            input_ch = 2 * in_ch if has_skip else in_ch
            self.conv1 = nn.Conv2d(input_ch, out_ch, 3, padding=1)
            self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, stride=2, padding=1)
        else:
            self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
            self.transform = nn.Conv2d(out_ch, out_ch, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bnorm1 = nn.BatchNorm2d(out_ch)
        self.bnorm2 = nn.BatchNorm2d(out_ch)
        self.relu  = nn.ReLU()
        
    def forward(self, x, t):
        h = self.bnorm1(self.relu(self.conv1(x)))
        time_emb = self.relu(self.time_mlp(t))
        time_emb = time_emb[(..., ) + (None, ) * 2]
        h = h + time_emb
        h = self.bnorm2(self.relu(self.conv2(h)))
        return self.transform(h)

class ConditionalUNet(nn.Module):
    def __init__(self, in_channels=8, out_channels=8, time_emb_dim=128, condition_dim=512):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU()
        )
        self.cond_mlp = nn.Sequential(
            nn.Linear(condition_dim, time_emb_dim),
            nn.ReLU()
        )

        # 16x16 -> 8x8 -> 4x4
        self.down1 = Block(in_channels, 64, time_emb_dim)
        self.down2 = Block(64, 128, time_emb_dim)
        
        # 4x4 -> 8x8 -> 16x16
        # up1 receives h2 (128 channels) directly from down2, no skip cat yet.
        self.up1 = Block(128, 64, time_emb_dim, up=True, has_skip=False)
        # up2 receives cat(up1_out, h1) -> 64 + 64 = 128 channels. 
        # So in_ch=64, and 2*in_ch = 128.
        self.up2 = Block(64, in_channels, time_emb_dim, up=True, has_skip=True)

        self.final_conv = nn.Conv2d(in_channels, out_channels, 1)

    def forward(self, x, t, condition):
        t = self.time_mlp(t)
        c = self.cond_mlp(condition)
        t_c = t + c

        # Down
        h1 = self.down1(x, t_c)   # 8x8
        h2 = self.down2(h1, t_c)  # 4x4
        
        # Up
        x = self.up1(h2, t_c)     # 8x8
        x = torch.cat((x, h1), dim=1) # Note: Block up expects 2*in_ch
        # Oops, my Block logic for Up is 2*in_ch. 
        # Let's fix the cat/up flow.
        # Actually my up1 input is 128, which matches down2 output.
        # But h1 is 64. So cat(up1_out, h1) is 64+64=128.
        # Let's make it cleaner.
        
        # Re-implement forward for clarity
        return self.refined_forward(x, t_c, h1, h2)

    def refined_forward(self, x, t_c, h1, h2):
        # Already did down
        # h1: [B, 64, 8, 8]
        # h2: [B, 128, 4, 4]
        
        x = self.up1(h2, t_c) # Output: [B, 64, 8, 8]
        x = torch.cat((x, h1), dim=1) # [B, 128, 8, 8]
        
        # Wait, Block(up=True) for up2 needs to handle the cat.
        # My Block init: self.conv1 = nn.Conv2d(2*in_ch, out_ch, ...)
        # So up2 should be Block(64, in_channels, ..., up=True) -> 2*64 = 128 input. Correct.
        
        x = self.up2(x, t_c) # Output: [B, in_channels, 16, 16]
        return self.final_conv(x)
