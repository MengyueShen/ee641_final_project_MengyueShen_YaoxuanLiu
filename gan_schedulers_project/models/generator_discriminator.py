import math
from typing import Tuple

import torch
import torch.nn as nn


class ConditionalGenerator(nn.Module):
    def __init__(
        self,
        noise_dim: int = 128,
        text_dim: int = 512,
        base_channels: int = 64,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        assert image_size in [64, 128, 256]
        self.noise_dim = noise_dim
        self.text_dim = text_dim
        self.image_size = image_size

        start_size = 4
        num_ups = int(math.log2(image_size) - math.log2(start_size))

        self.fc = nn.Linear(noise_dim + text_dim, base_channels * 2 ** num_ups * 4 * 4)

        ups = []
        in_ch = base_channels * 2 ** num_ups
        for _ in range(num_ups):
            out_ch = max(base_channels, in_ch // 2)
            ups.append(
                nn.Sequential(
                    nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch),
                    nn.ReLU(True),
                )
            )
            in_ch = out_ch

        self.ups = nn.Sequential(*ups)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(in_ch, 3, 3, 1, 1),
            nn.Tanh(),
        )

    def forward(self, noise: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        x = torch.cat([noise, text_features], dim=1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 4, 4)
        x = self.ups(x)
        x = self.to_rgb(x)
        return x


class ConditionalDiscriminator(nn.Module):
    def __init__(
        self,
        text_dim: int = 512,
        base_channels: int = 64,
        image_size: int = 256,
    ) -> None:
        super().__init__()
        assert image_size in [64, 128, 256]
        self.image_size = image_size
        start_size = 4
        num_downs = int(math.log2(image_size) - math.log2(start_size))

        downs = []
        in_ch = 3
        for i in range(num_downs):
            out_ch = base_channels * 2 ** i
            downs.append(
                nn.Sequential(
                    nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                    nn.BatchNorm2d(out_ch) if i > 0 else nn.Identity(),
                    nn.LeakyReLU(0.2, inplace=True),
                )
            )
            in_ch = out_ch

        self.downs = nn.Sequential(*downs)

        self.text_proj = nn.Linear(text_dim, in_ch)
        self.final = nn.Conv2d(in_ch, 1, 4, 1, 0)

    def forward(self, images: torch.Tensor, text_features: torch.Tensor) -> torch.Tensor:
        x = self.downs(images)
        text_embed = self.text_proj(text_features)
        text_embed = text_embed.view(text_embed.size(0), text_embed.size(1), 1, 1)
        text_embed = text_embed.expand_as(x)
        x = x + text_embed
        x = self.final(x)
        x = x.view(x.size(0), 1)
        return x
