from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


def _activation_from_name(name: str):
    if name == "gelu":
        return F.gelu
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
