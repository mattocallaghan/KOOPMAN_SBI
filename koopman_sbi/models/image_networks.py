"""Convolutional networks for tasks whose parameters and observations are square single-channel images.

All modules keep the flat-vector interface of DenseResidualNet (images are flattened to side * side), so the
models only choose a different network class; the Koopman operator between encoder and decoder is unchanged.
`hidden_dims` are channel widths, one per resolution level (each level halves the side length).
"""

from __future__ import annotations

import math
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def image_side(num_pixels: int) -> int:
    side = int(round(math.sqrt(num_pixels)))
    if side * side != num_pixels:
        raise ValueError(f"Image networks need a square single-channel image; got {num_pixels} values.")
    return side


def _linear(in_features: int, out_features: int, rank: int = 0) -> nn.Module:
    """Dense map, or (rank > 0) factored through `rank` dimensions: in -> rank -> out (fewer parameters)."""
    if rank <= 0 or rank >= min(in_features, out_features):
        return nn.Linear(in_features, out_features)
    return nn.Sequential(nn.Linear(in_features, rank, bias=False), nn.Linear(rank, out_features))


def _group_norm(channels: int) -> nn.GroupNorm:
    return nn.GroupNorm(num_groups=min(8, channels), num_channels=channels)


class _ResBlock(nn.Module):
    """Conv residual block with an optional additive conditioning vector (e.g. a time embedding)."""

    def __init__(self, in_channels: int, out_channels: int, cond_dim: int = 0) -> None:
        super().__init__()
        self.norm1 = _group_norm(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.norm2 = _group_norm(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.cond = nn.Linear(cond_dim, out_channels) if cond_dim > 0 else None
        self.skip = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()

    def forward(self, x: torch.Tensor, cond: torch.Tensor | None = None) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        if self.cond is not None and cond is not None:
            h = h + self.cond(cond)[:, :, None, None]
        h = self.conv2(F.silu(self.norm2(h)))
        return h + self.skip(x)


class _TimeEmbedding(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.register_buffer("frequencies", torch.exp(torch.linspace(0.0, math.log(1000.0), dim // 2)))
        self.mlp = nn.Sequential(nn.Linear(dim, dim), nn.SiLU(), nn.Linear(dim, dim))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        angles = time.reshape(-1, 1) * self.frequencies[None, :]
        return self.mlp(torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1))


class ConditionalConvUNet(nn.Module):
    """Vector field v(t, theta, x) for image parameters and image observations.

    Input is the flat concatenation [theta (side^2), context (side^2), time (1)], as for the dense network.
    theta and the observation are stacked as two channels; time enters every block through an embedding.
    """

    def __init__(self, input_dim: int, context_dim: int, hidden_dims: List[int]) -> None:
        super().__init__()
        if context_dim != input_dim:
            raise ValueError("ConvUNet expects the observation to be an image of the same size as theta.")
        self.input_dim = input_dim
        self.side = image_side(input_dim)
        channels = list(hidden_dims)
        cond_dim = 4 * channels[0]
        self.time_embedding = _TimeEmbedding(cond_dim)
        self.stem = nn.Conv2d(2, channels[0], 3, padding=1)
        self.down_blocks = nn.ModuleList()
        self.downsample = nn.ModuleList()
        previous = channels[0]
        for index, width in enumerate(channels):
            self.down_blocks.append(_ResBlock(previous, width, cond_dim))
            previous = width
            if index < len(channels) - 1:
                self.downsample.append(nn.Conv2d(width, width, 3, stride=2, padding=1))
        self.middle = _ResBlock(previous, previous, cond_dim)
        self.up_blocks = nn.ModuleList()
        for width in reversed(channels):
            self.up_blocks.append(_ResBlock(previous + width, width, cond_dim))
            previous = width
        self.head = nn.Sequential(_group_norm(previous), nn.SiLU(), nn.Conv2d(previous, 1, 3, padding=1))
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, model_input: torch.Tensor) -> torch.Tensor:
        n = self.input_dim
        theta, context, time = model_input[:, :n], model_input[:, n:2 * n], model_input[:, 2 * n:]
        h = self.stem(torch.stack([theta, context], dim=1).reshape(-1, 2, self.side, self.side))
        cond = self.time_embedding(time)
        skips = []
        for index, block in enumerate(self.down_blocks):
            h = block(h, cond)
            skips.append(h)
            if index < len(self.downsample):
                h = self.downsample[index](h)
        h = self.middle(h, cond)
        for block in self.up_blocks:
            skip = skips.pop()
            if h.shape[-1] != skip.shape[-1]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            h = block(torch.cat([h, skip], dim=1), cond)
        return self.head(h).reshape(-1, n)


class ConvEncoder(nn.Module):
    """Image (optionally with leading scalar inputs, e.g. time) -> feature vector.

    input_dim = extra + side^2: the first `extra` values are broadcast as constant channels.
    With downsample=False every level stays at full resolution (no stride-2 steps), so no per-pixel detail is
    pooled away before the linear map to the feature vector; keep the last width small (e.g. 4) to bound it.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        num_pixels: int,
        downsample: bool = True,
        projection_rank: int = 0,
    ) -> None:
        super().__init__()
        self.side = image_side(num_pixels)
        self.num_extra = input_dim - num_pixels
        if self.num_extra < 0:
            raise ValueError("ConvEncoder input is smaller than the image.")
        layers: List[nn.Module] = []
        previous = 1 + self.num_extra
        side = self.side
        for index, width in enumerate(hidden_dims):
            stride = 2 if (downsample and index > 0) else 1
            layers += [nn.Conv2d(previous, width, 3, stride=stride, padding=1), _group_norm(width), nn.SiLU()]
            previous = width
            side = side if stride == 1 else (side + 1) // 2
        self.features = nn.Sequential(*layers)
        self.head = _linear(previous * side * side, output_dim, projection_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        extra, pixels = x[:, :self.num_extra], x[:, self.num_extra:]
        image = pixels.reshape(-1, 1, self.side, self.side)
        if self.num_extra:
            image = torch.cat([image, extra[:, :, None, None].expand(-1, -1, self.side, self.side)], dim=1)
        return self.head(self.features(image).flatten(1))


class ConvDecoder(nn.Module):
    """Feature vector -> image (flat). hidden_dims run from the coarsest to the finest resolution.

    With downsample=False the latent is projected straight to hidden_dims[0] channels at full resolution and all
    blocks stay there (no upsampling); keep hidden_dims[0] small (e.g. 4) to bound the projection.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        downsample: bool = True,
        projection_rank: int = 0,
    ) -> None:
        super().__init__()
        self.side = image_side(output_dim)
        self.levels = len(hidden_dims)
        self.base_side = self.side
        for _ in range(self.levels - 1 if downsample else 0):
            self.base_side = (self.base_side + 1) // 2
        self.base_channels = hidden_dims[0]
        self.project = _linear(input_dim, hidden_dims[0] * self.base_side * self.base_side, projection_rank)
        self.blocks = nn.ModuleList()
        previous = hidden_dims[0]
        for width in hidden_dims:
            self.blocks.append(_ResBlock(previous, width))
            previous = width
        self.head = nn.Sequential(_group_norm(previous), nn.SiLU(), nn.Conv2d(previous, 1, 3, padding=1))
        self._sizes = []
        side = self.side
        for _ in range(self.levels):
            self._sizes.append(side)
            side = (side + 1) // 2 if downsample else side
        self._sizes.reverse()

    def forward(self, latent: torch.Tensor) -> torch.Tensor:
        h = self.project(latent).reshape(-1, self.base_channels, self.base_side, self.base_side)
        for block, size in zip(self.blocks, self._sizes):
            if h.shape[-1] != size:
                h = F.interpolate(h, size=(size, size), mode="nearest")
            h = block(h)
        return self.head(h).reshape(latent.shape[0], -1)
