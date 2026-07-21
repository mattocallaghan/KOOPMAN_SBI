from __future__ import annotations

import math
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from koopman_sbi.config import CMPEModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


class ConsistencyModelPosteriorEstimator(BasePosteriorModel):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: CMPEModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        network_cfg = model_config.network
        self.student = DenseResidualNet(
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
        self.current_step = 0
        self.total_training_steps = 1
        sigma2 = torch.full(
            (1, input_dim),
            float(model_config.sigma_data) ** 2,
            dtype=torch.float32,
            device=device,
        )
        self.register_buffer("sigma2", sigma2)

    def set_total_training_steps(self, total_training_steps: int) -> None:
        self.total_training_steps = max(int(total_training_steps), 1)

    def set_sigma2_from_theta(self, theta: torch.Tensor) -> None:
        theta = theta.detach().to(self.device)
        sigma2 = torch.var(theta, dim=0, keepdim=True, unbiased=False)
        sigma2 = torch.clamp(sigma2, min=1e-12)
        self.sigma2.copy_(sigma2)

    def sample_base_noise(self, batch_size: int, std: float = 1.0) -> torch.Tensor:
        return std * torch.randn(batch_size, self.input_dim, device=self.device)

    def _student_forward(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.dim() == 1:
            time = time.unsqueeze(1)
        model_input = torch.cat([x, context, time], dim=-1)
        return self.student(model_input)

    def consistency_function(self, x: torch.Tensor, context: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        if time.dim() == 1:
            time = time.unsqueeze(1)
        network_out = self._student_forward(x, context, time)
        sigma2 = self.sigma2.to(x.device)
        eps = float(self.model_config.eps)
        cskip = sigma2 / (((time - eps) ** 2) + sigma2)
        sigma = torch.sqrt(sigma2)
        cout = sigma * (time - eps) / torch.sqrt(sigma2 + time**2)
        return cskip * x + cout * network_out

    def _schedule_discretization(self) -> int:
        s0 = float(self.model_config.s0)
        s1 = float(self.model_config.s1)
        log_ratio = math.log(max(s1 / s0, 1.0), 2.0) if s1 > s0 else 0.0
        k_prime = max(int(math.floor(self.total_training_steps / (log_ratio + 1.0))), 1)
        value = min(s0 * (2.0 ** math.floor(self.current_step / k_prime)), s1) + 1.0
        return int(value)

    def _discretize_time(self, num_steps: int) -> torch.Tensor:
        eps = float(self.model_config.eps)
        t_max = float(self.model_config.t_max)
        rho = float(self.model_config.rho)
        steps = torch.arange(1, num_steps + 2, dtype=torch.float32, device=self.device)
        n_value = float(num_steps) + 1.0
        one_over_rho = 1.0 / rho
        return (
            eps**one_over_rho
            + ((steps - 1.0) / (n_value - 1.0)) * (t_max**one_over_rho - eps**one_over_rho)
        ) ** rho

    def _sample_neighboring_times(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        n_current = self._schedule_discretization()
        discretized_time = self._discretize_time(n_current)
        p_mean = float(self.model_config.p_mean)
        p_std = float(self.model_config.p_std)
        sqrt_two = math.sqrt(2.0)
        log_t_high = torch.log(discretized_time[1:])
        log_t_low = torch.log(discretized_time[:-1])
        erf_high = torch.erf((log_t_high - p_mean) / (sqrt_two * p_std))
        erf_low = torch.erf((log_t_low - p_mean) / (sqrt_two * p_std))
        probs = torch.clamp(erf_high - erf_low, min=1e-12)
        probs = probs / probs.sum()
        indices = torch.multinomial(probs, num_samples=batch_size, replacement=True)
        t1 = discretized_time[indices].unsqueeze(1)
        t2 = discretized_time[indices + 1].unsqueeze(1)
        return t1, t2

    def _pseudo_huber(self, difference: torch.Tensor) -> torch.Tensor:
        c_huber = 0.00054 * math.sqrt(float(self.input_dim))
        c_huber2 = c_huber * c_huber
        return torch.sqrt(difference.square() + c_huber2) - c_huber

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        theta_target, context = batch
        batch_size = theta_target.shape[0]
        self.current_step += 1
        z = self.sample_base_noise(batch_size)
        t1, t2 = self._sample_neighboring_times(batch_size)
        x_t1 = theta_target + t1 * z
        x_t2 = theta_target + t2 * z
        with torch.no_grad():
            teacher_out = self.consistency_function(x_t1, context, t1)
        student_out = self.consistency_function(x_t2, context, t2)
        lam = 1.0 / torch.clamp(t2 - t1, min=1e-12)
        loss = lam * self._pseudo_huber(teacher_out - student_out)
        consistency_loss = loss.mean()
        return {
            "total_loss": consistency_loss,
            "consistency_loss": consistency_loss,
        }

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            context = move_tensor_to_device(context, self.device)
            batch_size = context.shape[0]
            step_count = int(num_steps or self.model_config.default_num_steps)
            if step_count < 1:
                raise ValueError("cmpe.sample_batch requires num_steps >= 1.")

            discretized_time = torch.flip(self._discretize_time(step_count), dims=[0])
            initial_std = float(self.model_config.t_max)
            z_init = initial_noise if initial_noise is not None else self.sample_base_noise(batch_size, std=initial_std)
            z_init = move_tensor_to_device(z_init, self.device)
            samples = self.consistency_function(
                z_init,
                context,
                discretized_time[0].expand(batch_size).unsqueeze(1),
            )
            eps = float(self.model_config.eps)
            for index in range(1, len(discretized_time)):
                t_value = discretized_time[index]
                z = self.sample_base_noise(batch_size)
                noise_scale = math.sqrt(max(float(t_value * t_value) - eps * eps, 0.0))
                x_n = samples + noise_scale * z
                samples = self.consistency_function(
                    x_n,
                    context,
                    t_value.expand(batch_size).unsqueeze(1),
                )
            return samples

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "eps": self.model_config.eps,
                    "t_max": self.model_config.t_max,
                    "rho": self.model_config.rho,
                    "sigma_data": self.model_config.sigma_data,
                    "sigma2": self.sigma2.detach().cpu(),
                    "s0": self.model_config.s0,
                    "s1": self.model_config.s1,
                    "p_mean": self.model_config.p_mean,
                    "p_std": self.model_config.p_std,
                    "default_num_steps": self.model_config.default_num_steps,
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
    def load(cls, filepath: str, device: torch.device) -> "ConsistencyModelPosteriorEstimator":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = CMPEModelConfig(
            eps=model_config_dict.get("eps", 1e-3),
            t_max=model_config_dict.get("t_max", 200.0),
            rho=model_config_dict.get("rho", 7.0),
            sigma_data=model_config_dict.get("sigma_data", 1.0),
            s0=model_config_dict.get("s0", 10),
            s1=model_config_dict.get("s1", 50),
            p_mean=model_config_dict.get("p_mean", -1.1),
            p_std=model_config_dict.get("p_std", 2.0),
            default_num_steps=model_config_dict.get("default_num_steps", 10),
            network=NetworkConfig(**model_config_dict["network"]),
        )
        model = cls(
            input_dim=checkpoint["input_dim"],
            context_dim=checkpoint["context_dim"],
            model_config=model_config,
            device=device,
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        if "sigma2" in model_config_dict:
            sigma2_tensor = model_config_dict["sigma2"].to(device=device, dtype=torch.float32)
            model.sigma2.copy_(sigma2_tensor)
        model.to(device)
        return model
