from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

try:
    from torchdiffeq import odeint
except ImportError:
    def odeint(func, y0, t, atol=None, rtol=None, method=None, options=None):
        del atol, rtol, options
        states = [y0]
        current = y0
        for index in range(1, len(t)):
            t_prev = t[index - 1]
            t_next = t[index]
            dt = t_next - t_prev
            if method == "rk4":
                k1 = func(t_prev, current)
                k2 = func(t_prev + 0.5 * dt, current + 0.5 * dt * k1)
                k3 = func(t_prev + 0.5 * dt, current + 0.5 * dt * k2)
                k4 = func(t_next, current + dt * k3)
                current = current + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)
            else:
                current = current + dt * func(t_prev, current)
            states.append(current)
        return torch.stack(states, dim=0)

from koopman_sbi.config import FlowMatchingModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


class ConditionalFlowMatching(BasePosteriorModel):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: FlowMatchingModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        network_cfg = model_config.network
        self.vector_field = DenseResidualNet(
            input_dim=input_dim + context_dim + 1,
            output_dim=input_dim,
            hidden_dims=network_cfg.hidden_dims,
            activation=network_cfg.activation,
            batch_norm=network_cfg.batch_norm,
            dropout=network_cfg.dropout,
            theta_dim=input_dim,
            context_dim=context_dim,
            time_dim=1,
            theta_with_glu=network_cfg.theta_with_glu,
            context_with_glu=network_cfg.context_with_glu,
        )

    def sample_time(self, batch_size: int) -> torch.Tensor:
        exponent = self.model_config.time_prior_exponent
        uniform = torch.rand(batch_size, device=self.device)
        if exponent == 1:
            return uniform
        return uniform ** (1.0 / exponent)

    def sample_base_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.input_dim, device=self.device)

    def interpolate_state(
        self,
        noise_state: torch.Tensor,
        target_state: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        sigma_min = self.model_config.sigma_min
        return (1 - (1 - sigma_min) * time)[:, None] * noise_state + time[:, None] * target_state

    def forward(self, time: torch.Tensor, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        time = time * torch.ones(len(theta), device=theta.device)
        if time.dim() == 1:
            time = time.unsqueeze(1)
        model_input = torch.cat([theta, context, time], dim=-1)
        return self.vector_field(model_input)

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        theta_target, context = batch
        batch_size = theta_target.shape[0]
        time = self.sample_time(batch_size)
        noise_state = self.sample_base_noise(batch_size)
        interpolated = self.interpolate_state(noise_state, theta_target, time)
        true_velocity = theta_target - (1 - self.model_config.sigma_min) * noise_state
        predicted_velocity = self.forward(time, interpolated, context)
        flow_matching_loss = nn.MSELoss()(predicted_velocity, true_velocity)
        return {
            "total_loss": flow_matching_loss,
            "flow_matching_loss": flow_matching_loss,
        }

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
        integration_steps: Optional[int] = None,
        solver: Optional[str] = None,
        atol: Optional[float] = None,
        rtol: Optional[float] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            context = move_tensor_to_device(context, self.device)
            batch_size = context.shape[0]
            theta_0 = initial_noise if initial_noise is not None else self.sample_base_noise(batch_size)
            theta_0 = move_tensor_to_device(theta_0, self.device)
            dtype = torch.float32 if self.device.type == "mps" else torch.float64
            integration_steps = self.model_config.integration_steps if integration_steps is None else integration_steps
            if integration_steps is not None and int(integration_steps) < 2:
                raise ValueError("flow_matching.integration_steps must be at least 2 when provided.")
            if integration_steps is None:
                t_span = torch.tensor([0.0, 1.0 - self.model_config.sigma_min], dtype=dtype, device=self.device)
                method = solver or "dopri5"
            else:
                t_span = torch.linspace(
                    0.0,
                    1.0 - self.model_config.sigma_min,
                    steps=int(integration_steps),
                    dtype=dtype,
                    device=self.device,
                )
                method = solver or "rk4"
            ode_options = {"dtype": torch.float32} if self.device.type == "mps" else {}
            trajectory = odeint(
                lambda current_t, theta_t: self.forward(current_t, theta_t, context),
                theta_0,
                t_span,
                atol=self.model_config.atol if atol is None else atol,
                rtol=self.model_config.rtol if rtol is None else rtol,
                method=method,
                options=ode_options,
            )
            return trajectory[-1]

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "sigma_min": self.model_config.sigma_min,
                    "time_prior_exponent": self.model_config.time_prior_exponent,
                    "integration_steps": self.model_config.integration_steps,
                    "atol": self.model_config.atol,
                    "rtol": self.model_config.rtol,
                    "network": {
                        "hidden_dims": self.model_config.network.hidden_dims,
                        "activation": self.model_config.network.activation,
                        "batch_norm": self.model_config.network.batch_norm,
                        "dropout": self.model_config.network.dropout,
                        "theta_with_glu": self.model_config.network.theta_with_glu,
                        "context_with_glu": self.model_config.network.context_with_glu,
                        "type": self.model_config.network.type,
                    },
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: torch.device) -> "ConditionalFlowMatching":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = FlowMatchingModelConfig(
            sigma_min=model_config_dict["sigma_min"],
            time_prior_exponent=model_config_dict["time_prior_exponent"],
            integration_steps=model_config_dict.get("integration_steps"),
            atol=model_config_dict.get("atol", 1e-5),
            rtol=model_config_dict.get("rtol", 1e-5),
            network=NetworkConfig(**model_config_dict["network"]),
        )
        model = cls(
            input_dim=checkpoint["input_dim"],
            context_dim=checkpoint["context_dim"],
            model_config=model_config,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(device)
        return model
