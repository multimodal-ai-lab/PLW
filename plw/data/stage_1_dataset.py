import os
from glob import glob
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def generate_random_bit_message(size):
    message = np.random.binomial(1, 0.5, size)
    message = torch.from_numpy(message).float()
    return message


class Stage1ImageDataset(Dataset):

    def __init__(
        self, data_path, message_size=48, img_size=(512, 512), num_samples=None
    ):
        self.data_path = data_path
        self.files_list = sorted(
            glob(os.path.join(self.data_path, "*.png"))
            + glob(os.path.join(self.data_path, "*.jpg"))
        )
        if num_samples is not None:
            self.files_list = self.files_list[:num_samples]
        self.message_size = message_size
        self.img_size = img_size

    def __getitem__(self, idx):
        img_cover_path = self.files_list[idx]
        img_cover = Image.open(img_cover_path).convert("RGB")
        transform_pipeline = transforms.Compose(
            [
                transforms.Resize(
                    self.img_size, interpolation=transforms.InterpolationMode.BILINEAR
                ),
                transforms.ToTensor(),
            ]
        )
        img_cover = transform_pipeline(img_cover)
        message = generate_random_bit_message(size=self.message_size)
        return (img_cover, message)

    def __len__(self):
        return len(self.files_list)
