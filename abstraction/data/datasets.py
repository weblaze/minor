import os

import torchvision.transforms as transforms
from PIL import Image, ImageOps
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_path, size=128, train=True):
        self.image_path = image_path
        self.image_files = [f for f in os.listdir(image_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        self.train = train
        self.transform = transforms.Compose([
            transforms.Resize((size, size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_path, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        image = ImageOps.exif_transpose(image)
        image = self.transform(image)
        return image
