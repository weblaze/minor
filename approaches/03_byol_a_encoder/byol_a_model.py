"""BYOL-A: self-supervised audio encoder (Niizumi et al. 2021, simplified).

Two augmented views of the same log-mel spectrogram are encoded; the online
network predicts the target network's projection; the target is an EMA of the
online weights. No negative pairs needed. Augmentations follow the paper:
log-mixup-exp with a memory bank of past samples, plus random resize crop.

Kept approach-local until the ablation against CLAP decides whether it gets
promoted into abstraction/audio/.
"""
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

N_MELS = 64
TARGET_FRAMES = 96  # ~1s crops at hop 160 / sr 16000


class AudioEncoder(nn.Module):
    """CNN over log-mel spectrograms -> embedding (default 512-d to match CLAP)."""

    def __init__(self, embed_dim=512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 256, 3, stride=1, padding=1), nn.BatchNorm2d(256), nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), nn.BatchNorm2d(512), nn.ReLU(),
        )
        self.fc = nn.Linear(512, embed_dim)

    def forward(self, x):              # x: [B, 1, N_MELS, T]
        h = self.conv(x)
        h = h.mean(dim=(2, 3))         # global average pool
        return self.fc(h)


def mlp(in_dim, hidden_dim, out_dim):
    return nn.Sequential(
        nn.Linear(in_dim, hidden_dim), nn.BatchNorm1d(hidden_dim), nn.ReLU(),
        nn.Linear(hidden_dim, out_dim),
    )


class BYOLA(nn.Module):
    """Online encoder+projector+predictor vs EMA target encoder+projector."""

    def __init__(self, embed_dim=512, proj_dim=256, hidden_dim=1024, ema_decay=0.99):
        super().__init__()
        self.ema_decay = ema_decay
        self.online_encoder = AudioEncoder(embed_dim)
        self.online_projector = mlp(embed_dim, hidden_dim, proj_dim)
        self.predictor = mlp(proj_dim, hidden_dim, proj_dim)

        self.target_encoder = AudioEncoder(embed_dim)
        self.target_projector = mlp(embed_dim, hidden_dim, proj_dim)
        self._copy_weights()
        for p in self.target_encoder.parameters():
            p.requires_grad = False
        for p in self.target_projector.parameters():
            p.requires_grad = False

    def _copy_weights(self):
        self.target_encoder.load_state_dict(self.online_encoder.state_dict())
        self.target_projector.load_state_dict(self.online_projector.state_dict())

    @torch.no_grad()
    def update_target(self):
        for online, target in (
            (self.online_encoder, self.target_encoder),
            (self.online_projector, self.target_projector),
        ):
            for p_o, p_t in zip(online.parameters(), target.parameters()):
                p_t.data = self.ema_decay * p_t.data + (1 - self.ema_decay) * p_o.data

    @staticmethod
    def regression_loss(pred, target):
        pred = F.normalize(pred, dim=-1)
        target = F.normalize(target, dim=-1)
        return 2 - 2 * (pred * target).sum(dim=-1).mean()

    def forward(self, view1, view2):
        loss = 0
        for a, b in ((view1, view2), (view2, view1)):
            pred = self.predictor(self.online_projector(self.online_encoder(a)))
            with torch.no_grad():
                target = self.target_projector(self.target_encoder(b))
            loss = loss + self.regression_loss(pred, target)
        return loss / 2


class Augmenter:
    """BYOL-A augmentations on per-sample-normalized log-mel: mixup + random resize crop."""

    def __init__(self, memory_size=2048, mixup_alpha=0.4):
        self.memory = []
        self.memory_size = memory_size
        self.mixup_alpha = mixup_alpha

    def mixup(self, x):
        # log-mixup-exp: mix in the linear domain, return to log
        if self.memory:
            past = random.choice(self.memory)
            lam = random.uniform(0, self.mixup_alpha)
            x = torch.log((1 - lam) * x.exp() + lam * past.exp() + 1e-8)
        self.memory.append(x.detach().clone())
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        return x

    @staticmethod
    def random_resize_crop(x, min_scale=0.6):
        # x: [1, F, T] — crop a random time-frequency box, resize back
        _, f, t = x.shape
        scale_f = random.uniform(min_scale, 1.0)
        scale_t = random.uniform(min_scale, 1.0)
        crop_f, crop_t = max(8, int(f * scale_f)), max(8, int(t * scale_t))
        top = random.randint(0, f - crop_f)
        left = random.randint(0, t - crop_t)
        crop = x[:, top:top + crop_f, left:left + crop_t]
        return F.interpolate(crop.unsqueeze(0), size=(f, t), mode="bilinear",
                             align_corners=False).squeeze(0)

    def __call__(self, x):
        return self.random_resize_crop(self.mixup(x))
