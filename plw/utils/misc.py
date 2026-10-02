from __future__ import annotations
from dataclasses import asdict
from typing import TYPE_CHECKING
import numpy as np
import torch
import yaml

if TYPE_CHECKING:
    from plw.wrappers.bagel.modeling.autoencoder import AutoEncoder


def tensor_to_latent(x: torch.Tensor, vae: AutoEncoder) -> torch.Tensor:
    x = 2.0 * x - 1.0
    latents = vae.encode(x)
    return latents


def latent_to_tensor(latents: torch.Tensor, vae: AutoEncoder) -> torch.Tensor:
    image = vae.decode(latents)
    image_tensor = image / 2.0 + 0.5
    return image_tensor


def random_float(min_val: float, max_val: float) -> float:
    return np.random.rand() * (max_val - min_val) + min_val


def save_config(config, path: str) -> None:
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)
