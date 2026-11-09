import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import Dinov2Config, Dinov2Model

DINO_ID = "facebook/dinov2-small"
DIM = 384
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class DinoFeatures(nn.Module):
    """Frozen DINOv2 ViT-S/14. Differentiable with respect to the input images.

    Input images are in [-1, 1]. They are resized to `size` (a multiple of 14)
    and normalised with the statistics DINOv2 was trained with. `pooling` is
    "cls", "patch" (mean of patch tokens) or "tokens" (all tokens, unpooled).

    `pretrained=False` builds a randomly initialised network of the same shape.
    It exists for smoke tests without a download and must not be used for results.
    """

    def __init__(self, size=112, pooling="cls", pretrained=True, depth=12):
        super().__init__()
        if size % 14:
            raise ValueError("size must be a multiple of 14")
        if pooling not in ("cls", "patch", "tokens"):
            raise ValueError(f"unknown pooling {pooling!r}")
        if pretrained:
            self.model = Dinov2Model.from_pretrained(DINO_ID)
        else:
            self.model = Dinov2Model(Dinov2Config(
                hidden_size=DIM, num_hidden_layers=depth, num_attention_heads=6,
                patch_size=14, image_size=518))
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad = False
        self.size = size
        self.pooling = pooling
        self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        return super().train(False)

    def forward(self, images, pooling=None):
        pooling = pooling or self.pooling
        x = (images + 1) / 2
        x = F.interpolate(x, size=(self.size, self.size), mode="bilinear",
                          align_corners=False, antialias=True)
        x = (x - self.mean) / self.std
        tokens = self.model(pixel_values=x).last_hidden_state
        if pooling == "cls":
            return tokens[:, 0]
        if pooling == "patch":
            return tokens[:, 1:].mean(dim=1)
        return tokens
