import io
import torch
from PIL import Image
from torch.nn import functional as F
from torchvision import transforms as T
import kornia as K


def augmentation(encoded_images: torch.Tensor, type: str) -> torch.Tensor:
    if type == "identity":
        images_aug = encoded_images
    elif type == "brightness":
        images_aug = K.augmentation.ColorJiggle(
            brightness=(0.8, 1.2),
            contrast=(1.0, 1.0),
            saturation=(1.0, 1.0),
            hue=(0.0, 0.0),
            p=1,
        )(encoded_images)
    elif type == "contrast":
        images_aug = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),
            contrast=(0.8, 1.2),
            saturation=(1.0, 1.0),
            hue=(0.0, 0.0),
            p=1,
        )(encoded_images)
    elif type == "saturation":
        images_aug = K.augmentation.ColorJiggle(
            brightness=(1.0, 1.0),
            contrast=(1.0, 1.0),
            saturation=(0.8, 1.2),
            hue=(0.0, 0.0),
            p=1,
        )(encoded_images)
    elif type == "blur":
        images_aug = K.augmentation.RandomGaussianBlur((3, 3), (4.0, 4.0), p=1.0)(
            encoded_images
        )
    elif type == "noise":
        images_aug = K.augmentation.RandomGaussianNoise(mean=0.0, std=0.1, p=1)(
            encoded_images
        )
    elif type == "jpeg_compress":
        B = encoded_images.shape[0]
        images_aug = []
        for i in range(B):
            buffer = io.BytesIO()
            pil_image = T.ToPILImage()(encoded_images[i].squeeze(0))
            pil_image.save(buffer, format="JPEG", quality=50)
            buffer.seek(0)
            pil_image = Image.open(buffer)
            images_aug.append(
                T.ToTensor()(pil_image).to(encoded_images.device).unsqueeze(0)
            )
        images_aug = torch.cat(images_aug, dim=0)
    elif type == "resize":
        images_aug = F.interpolate(
            encoded_images, scale_factor=(0.5, 0.5), mode="bilinear"
        )
    elif type == "sharpness":
        images_aug = K.augmentation.RandomSharpness(sharpness=10.0, p=1)(encoded_images)
    else:
        raise ValueError(f"Wrong augmentation type in augmentation(...)")
    return torch.clamp(images_aug, 0, 1)
