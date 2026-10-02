import numpy as np
from torch import nn
from plw.modeling.blocks import View, Repeat, conv_nd, Conv2D, Flatten, MLP


class MessageEncoder(nn.Module):

    def __init__(self, message_size, latent_channels, base_res=32, resolution=64):
        super().__init__()
        log_resolution = int(np.log2(resolution))
        log_base = int(np.log2(base_res))
        self.message_size = message_size
        self.message_scaler = nn.Sequential(
            nn.Linear(message_size, base_res * base_res),
            nn.SiLU(),
            nn.Linear(base_res * base_res, base_res * base_res),
            nn.SiLU(),
            View(-1, 1, base_res, base_res),
            Repeat(latent_channels, 1, 1),
            nn.Upsample(
                scale_factor=(
                    2 ** (log_resolution - log_base),
                    2 ** (log_resolution - log_base),
                )
            ),
            conv_nd(2, latent_channels, latent_channels, 3, padding=1),
        )

    def forward(self, sec):
        return self.message_scaler(sec)


class MessageExtractor(nn.Module):

    def __init__(self, latent_channels, message_size=48):
        super().__init__()
        self.decoder = nn.Sequential(
            Conv2D(latent_channels, 64, 3, strides=2, activation="selu"),
            Conv2D(64, 64, 3, activation="selu"),
            Conv2D(64, 128, 3, strides=2, activation="selu"),
            Conv2D(128, 128, 3, activation="selu"),
            Conv2D(128, 256, 3, strides=2, activation="selu"),
            Conv2D(256, 256, 3, activation="selu"),
            Conv2D(256, 512, 3, strides=2, activation="selu"),
            Conv2D(512, 512, 3, activation="selu"),
            Flatten(),
        )
        self.mlps = nn.Sequential(
            MLP(8192, 2048, activation="selu"),
            MLP(2048, 2048, activation="selu"),
            MLP(2048, 2048, activation="selu"),
            nn.Dropout(p=0.1),
            MLP(2048, message_size, activation=None),
        )

    def forward(self, latent):
        decoded = self.decoder(latent)
        decoded = self.mlps(decoded)
        return decoded
