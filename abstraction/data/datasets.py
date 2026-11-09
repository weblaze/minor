import os

import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset

from .splits import split_of

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg")


class ImageDataset(Dataset):
    """Images scaled to [-1, 1]. `split` restricts the set to "train", "val" or "test"
    by the hash of the file name; None keeps every file. Training applies a horizontal flip.
    """

    def __init__(self, image_path, size=128, train=True, split=None, val_percent=10, test_percent=10):
        self.image_path = image_path
        files = sorted(f for f in os.listdir(image_path) if f.lower().endswith(IMAGE_EXTENSIONS))
        if split is not None:
            files = [f for f in files if split_of(f, val_percent, test_percent) == split]
        self.image_files = files
        self.train = train
        steps = [transforms.Resize((size, size))]
        if train:
            steps.append(transforms.RandomHorizontalFlip())
        steps += [transforms.ToTensor(), transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3)]
        self.transform = transforms.Compose(steps)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_files[idx])
        image = ImageOps.exif_transpose(Image.open(img_path)).convert("RGB")
        return self.transform(image)
