from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _init_linear_he_normal(layer: nn.Linear) -> None:
    nn.init.kaiming_normal_(layer.weight, nonlinearity="relu")
    if layer.bias is not None:
        nn.init.zeros_(layer.bias)


def _activation_from_name(name: str):
    if name == "gelu":
        return F.gelu
    if name == "mish":
        return F.mish
    if name == "relu":
        return F.relu
    if name == "elu":
        return F.elu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation: {name}")


class ActivationLayer(nn.Module):
    def __init__(self, activation_name: str) -> None:
        super().__init__()
        self.activation = _activation_from_name(activation_name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x)


class DenseResidualNet(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "gelu",
        batch_norm: bool = False,
        dropout: float = 0.0,
        theta_dim: int = 0,
        context_dim: int = 0,
        time_dim: int = 0,
        theta_with_glu: bool = False,
        context_with_glu: bool = False,
    ) -> None:
        super().__init__()
        self.theta_dim = theta_dim
        self.context_dim = context_dim
        self.time_dim = time_dim
        self.theta_with_glu = theta_with_glu
        self.context_with_glu = context_with_glu

        if theta_with_glu:
            self.theta_glu = nn.Linear(theta_dim, 2 * theta_dim)
        if context_with_glu:
            self.context_glu = nn.Linear(context_dim, 2 * context_dim)

        layers = []
        current_dim = input_dim
        residual_flags = []
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(current_dim, hidden_dim))
            residual_flags.append(current_dim == hidden_dim)
            if batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(ActivationLayer(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            current_dim = hidden_dim
        layers.append(nn.Linear(current_dim, output_dim))

        self.layers = nn.ModuleList(layers)
        self.residual_flags = residual_flags

    def _apply_input_gates(self, x: torch.Tensor) -> torch.Tensor:
        if not self.theta_with_glu and not self.context_with_glu:
            return x

        start = 0
        theta = x[:, start : start + self.theta_dim]
        start += self.theta_dim
        context = x[:, start : start + self.context_dim]
        start += self.context_dim
        time = x[:, start : start + self.time_dim] if self.time_dim else x[:, 0:0]

        if self.theta_with_glu:
            theta = F.glu(self.theta_glu(theta), dim=-1)
        if self.context_with_glu:
            context = F.glu(self.context_glu(context), dim=-1)
        return torch.cat([theta, context, time], dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._apply_input_gates(x)
        residual_index = 0
        for layer in self.layers[:-1]:
            if isinstance(layer, nn.Linear):
                if residual_index < len(self.residual_flags) and self.residual_flags[residual_index]:
                    residual = x
                    x = layer(x)
                    x = x + residual
                else:
                    x = layer(x)
                residual_index += 1
            else:
                x = layer(x)
        return self.layers[-1](x)


class FourierEmbedding(nn.Module):
    def __init__(
        self,
        embed_dim: int = 32,
        scale: float = 30.0,
        include_identity: bool = True,
    ) -> None:
        super().__init__()
        if embed_dim % 2 != 0:
            raise ValueError(f"Embedding dimension must be even, got {embed_dim}.")
        self.embed_dim = embed_dim
        self.scale = scale
        self.include_identity = include_identity
        self.frequencies = nn.Parameter(torch.randn(embed_dim // 2))

    def forward(self, time: torch.Tensor) -> torch.Tensor:
        projection = time * self.frequencies[None, :] * (2.0 * torch.pi * self.scale)
        if self.include_identity:
            return torch.cat([time, torch.sin(projection), torch.cos(projection)], dim=-1)
        return torch.cat([torch.sin(projection), torch.cos(projection)], dim=-1)


class FiLM(nn.Module):
    def __init__(self, width: int, cond_dim: int, use_gamma: bool = False) -> None:
        super().__init__()
        self.use_gamma = use_gamma
        out_dim = 2 * width if use_gamma else width
        self.proj = nn.Linear(cond_dim, out_dim)
        _init_linear_he_normal(self.proj)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        modulation = self.proj(cond)
        if self.use_gamma:
            gamma, beta = modulation.chunk(2, dim=-1)
            return (1.0 + gamma) * hidden + beta
        return hidden + modulation


class ConditionalDenseBlock(nn.Module):
    def __init__(
        self,
        in_width: int,
        width: int,
        cond_dim: int,
        activation: str = "mish",
        residual: bool = True,
        dropout: float = 0.05,
        norm: str | None = "layer",
        film_use_gamma: bool = False,
    ) -> None:
        super().__init__()
        self.residual = residual
        self.dense = nn.Linear(in_width, width)
        _init_linear_he_normal(self.dense)
        self.activation = ActivationLayer(activation)
        self.dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()
        self.film = FiLM(width=width, cond_dim=cond_dim, use_gamma=film_use_gamma)
        if norm == "layer":
            self.norm = nn.LayerNorm(width)
        elif norm is None:
            self.norm = nn.Identity()
        else:
            raise ValueError(f"Unsupported normalization strategy: {norm}")
        self.projector = nn.Linear(in_width, width) if residual and in_width != width else None
        if self.projector is not None:
            _init_linear_he_normal(self.projector)

    def forward(self, hidden: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        updated = self.dense(hidden)
        updated = self.dropout(updated)
        updated = self.activation(updated)
        updated = self.film(updated, cond)
        if self.residual:
            skip = hidden if self.projector is None else self.projector(hidden)
            updated = skip + updated
        return self.norm(updated)


class TimeConditionedMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        condition_dim: int,
        widths: List[int],
        time_embedding_dim: int = 32,
        fourier_scale: float = 30.0,
        activation: str = "mish",
        residual: bool = True,
        dropout: float = 0.05,
        norm: str | None = "layer",
        merge: str = "concat",
        film_use_gamma: bool = False,
    ) -> None:
        super().__init__()
        if not widths:
            raise ValueError("TimeConditionedMLP requires at least one hidden width.")
        if merge not in ("add", "concat"):
            raise ValueError(f"Unknown merge mode: {merge!r}")
        self.widths = widths
        self.merge = merge
        self.time_embedding = (
            nn.Identity() if time_embedding_dim == 1 else FourierEmbedding(embed_dim=time_embedding_dim, scale=fourier_scale)
        )
        time_cond_dim = 1 if time_embedding_dim == 1 else time_embedding_dim + 1
        self.x_proj = nn.Linear(input_dim, widths[0])
        _init_linear_he_normal(self.x_proj)
        self.c_proj = nn.Linear(condition_dim, widths[0]) if condition_dim > 0 else None
        if self.c_proj is not None:
            _init_linear_he_normal(self.c_proj)
        merge_in_dim = widths[0] * 2 if merge == "concat" and self.c_proj is not None else widths[0]
        self.merge_proj = nn.Linear(merge_in_dim, widths[0]) if self.c_proj is not None else None
        if self.merge_proj is not None:
            _init_linear_he_normal(self.merge_proj)
        self.activation = ActivationLayer(activation)
        blocks = []
        current_width = widths[0]
        for width in widths:
            blocks.append(
                ConditionalDenseBlock(
                    in_width=current_width,
                    width=width,
                    cond_dim=time_cond_dim,
                    activation=activation,
                    residual=residual,
                    dropout=dropout,
                    norm=norm,
                    film_use_gamma=film_use_gamma,
                )
            )
            current_width = width
        self.blocks = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor, time: torch.Tensor, conditions: torch.Tensor | None) -> torch.Tensor:
        hidden = self.x_proj(x)
        if conditions is not None and self.c_proj is not None and self.merge_proj is not None:
            cond_hidden = self.c_proj(conditions)
            if self.merge == "concat":
                hidden = torch.cat([hidden, cond_hidden], dim=-1)
            else:
                hidden = hidden + cond_hidden
            hidden = self.merge_proj(self.activation(hidden))
        hidden = self.activation(hidden)
        time_emb = self.time_embedding(time)
        for block in self.blocks:
            hidden = block(hidden, time_emb)
        return hidden
