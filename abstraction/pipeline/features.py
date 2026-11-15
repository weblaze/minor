import torch
from torch.utils.data import DataLoader

from abstraction.encoders.dino import DinoFeatures


def build_dino(config, device, smoke):
    """Frozen DINOv2; randomly initialised with two blocks for smoke runs."""
    if smoke:
        return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"], pretrained=False, depth=2).to(device)
    return DinoFeatures(config["dino"]["size"], config["dino"]["pooling"]).to(device)


@torch.no_grad()
def features_of_dataset(dino, dataset, device, batch_size=64, transform=None):
    """DINOv2 features of every image of a dataset, optionally after `transform`."""
    out = []
    for images in DataLoader(dataset, batch_size=batch_size):
        images = images.to(device)
        if transform is not None:
            images = transform(images)
        out.append(dino(images).float().cpu())
    return torch.cat(out)
