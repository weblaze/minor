"""Frozen CAV-MAE encoder (audio and image branches, shared joint layers).

Re-implementation of the encoder part of `CAVMAE` from github.com/YuanGongND/cav-mae
(Gong et al., "Contrastive Audio-Visual Masked Autoencoder", ICLR 2023; BSD 2-Clause licence),
with parameter names equal to the published checkpoints so that their state dicts load directly.
The decoder of the masked autoencoder is not built; its checkpoint entries are ignored.

The model was pretrained on video (audio track and frames) without text or labels.
An input of either modality passes through its own transformer blocks and then through the joint
blocks with the layer normalisations of its modality. The embedding is the mean over tokens.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import Attention, Mlp

FBANK_MEAN, FBANK_STD = -5.081, 4.4849
AUDIO_SAMPLE_RATE = 16000
AUDIO_FRAMES, MEL_BINS = 1024, 128
IMAGE_SIZE = 224
IMAGE_MEAN, IMAGE_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)


class PatchEmbed(nn.Module):
    def __init__(self, in_chans, embed_dim, patch_size=16):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        return self.proj(x).flatten(2).transpose(1, 2)


class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1, self.norm1_a, self.norm1_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=True)
        self.norm2, self.norm2_a, self.norm2_v = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio))

    def forward(self, x, modality=None):
        suffix = "" if modality is None else f"_{modality}"
        x = x + self.attn(getattr(self, "norm1" + suffix)(x))
        return x + self.mlp(getattr(self, "norm2" + suffix)(x))


class CavMae(nn.Module):
    def __init__(self, embed_dim=768, num_heads=12, modality_specific_depth=11, total_depth=12,
                 audio_frames=AUDIO_FRAMES, image_size=IMAGE_SIZE):
        super().__init__()
        self.patch_embed_a = PatchEmbed(1, embed_dim)
        self.patch_embed_v = PatchEmbed(3, embed_dim)
        n_audio = (audio_frames // 16) * (MEL_BINS // 16)
        n_image = (image_size // 16) ** 2
        self.modality_a = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.modality_v = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed_a = nn.Parameter(0.02 * torch.randn(1, n_audio, embed_dim), requires_grad=False)
        self.pos_embed_v = nn.Parameter(0.02 * torch.randn(1, n_image, embed_dim), requires_grad=False)
        self.blocks_a = nn.ModuleList(Block(embed_dim, num_heads) for _ in range(modality_specific_depth))
        self.blocks_v = nn.ModuleList(Block(embed_dim, num_heads) for _ in range(modality_specific_depth))
        self.blocks_u = nn.ModuleList(Block(embed_dim, num_heads)
                                      for _ in range(total_depth - modality_specific_depth))
        self.norm_a, self.norm_v, self.norm = nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim), nn.LayerNorm(embed_dim)

    def embed_audio(self, fbank):
        """Normalised filterbank (b, frames, mel bins) to embeddings (b, embed_dim)."""
        x = self.patch_embed_a(fbank.unsqueeze(1).transpose(2, 3)) + self.pos_embed_a + self.modality_a
        for block in self.blocks_a:
            x = block(x)
        for block in self.blocks_u:
            x = block(x, "a")
        return self.norm_a(x).mean(dim=1)

    def embed_image(self, images):
        """Images (b, 3, 224, 224) with ImageNet normalisation to embeddings (b, embed_dim)."""
        x = self.patch_embed_v(images) + self.pos_embed_v + self.modality_v
        for block in self.blocks_v:
            x = block(x)
        for block in self.blocks_u:
            x = block(x, "v")
        return self.norm_v(x).mean(dim=1)


def load_cav_mae(checkpoint, device):
    """The frozen pretrained encoder. Every encoder parameter must be present in the checkpoint."""
    model = CavMae()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    state = {key.removeprefix("module."): value for key, value in state.items()}
    result = model.load_state_dict(state, strict=False)
    if result.missing_keys:
        raise RuntimeError(f"checkpoint lacks encoder parameters: {result.missing_keys[:5]}")
    unexpected = [key for key in result.unexpected_keys if not key.startswith(("decoder_", "mask_token"))]
    if unexpected:
        raise RuntimeError(f"unexpected checkpoint entries: {unexpected[:5]}")
    return model.eval().requires_grad_(False).to(device)


def smoke_cav_mae(device):
    """Randomly initialised small model with the input shapes of the pretrained one. For smoke runs only."""
    return CavMae(embed_dim=96, num_heads=4, modality_specific_depth=1, total_depth=2).eval().requires_grad_(False).to(device)


def fbank_of(waveform):
    """Mono waveform (samples,) at 16 kHz to the normalised filterbank (AUDIO_FRAMES, MEL_BINS)."""
    import torchaudio

    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform.unsqueeze(0), htk_compat=True, sample_frequency=AUDIO_SAMPLE_RATE, use_energy=False,
        window_type="hanning", num_mel_bins=MEL_BINS, dither=0.0, frame_shift=10)
    missing = AUDIO_FRAMES - fbank.shape[0]
    fbank = F.pad(fbank, (0, 0, 0, missing)) if missing > 0 else fbank[:AUDIO_FRAMES]
    return (fbank - FBANK_MEAN) / FBANK_STD
