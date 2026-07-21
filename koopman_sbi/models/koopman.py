from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from koopman_sbi.config import KoopmanModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


class KoopmanFlow(BasePosteriorModel):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: KoopmanModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        network_cfg = model_config.network

        self.encoder = DenseResidualNet(
            input_dim=input_dim + context_dim,
            output_dim=model_config.lifting_dim,
            hidden_dims=network_cfg.hidden_dims,
            activation=network_cfg.activation,
            batch_norm=network_cfg.batch_norm,
            dropout=network_cfg.dropout,
            theta_dim=input_dim,
            context_dim=context_dim,
            time_dim=0,
            theta_with_glu=network_cfg.theta_with_glu,
            context_with_glu=network_cfg.context_with_glu,
        )
        self.koopman_linear = nn.Linear(model_config.lifting_dim, model_config.lifting_dim)
        self.context_modulation = nn.Linear(context_dim, model_config.lifting_dim)
        self.decoder = DenseResidualNet(
            input_dim=model_config.lifting_dim + context_dim,
            output_dim=input_dim,
            hidden_dims=list(reversed(network_cfg.hidden_dims)),
            activation=network_cfg.activation,
            batch_norm=network_cfg.batch_norm,
            dropout=network_cfg.dropout,
            theta_dim=model_config.lifting_dim,
            context_dim=context_dim,
            time_dim=0,
            theta_with_glu=False,
            context_with_glu=network_cfg.context_with_glu,
        )

    def sample_base_noise(self, batch_size: int) -> torch.Tensor:
        return torch.randn(batch_size, self.input_dim, device=self.device)

    def encode_noise(self, noise_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([noise_state, context], dim=-1))

    def evolve_latent(self, latent_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.koopman_linear(latent_state) + self.context_modulation(context)

    def encode_target(self, target_theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.encoder(torch.cat([target_theta, context], dim=-1))

    def decode_latent(self, latent_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.decoder(torch.cat([latent_state, context], dim=-1))

    def forward(self, noise_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        lifted_noise = self.encode_noise(noise_state, context)
        evolved_latent = self.evolve_latent(lifted_noise, context)
        return self.decode_latent(evolved_latent, context)

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        noise_state, theta_target, context = batch
        lifted_noise = self.encode_noise(noise_state, context)
        evolved_latent = self.evolve_latent(lifted_noise, context)
        predicted_theta = self.decode_latent(evolved_latent, context)

        lifted_target = self.encode_target(theta_target, context)
        reconstructed_theta = self.decode_latent(lifted_target, context)

        prediction_loss = nn.MSELoss()(predicted_theta, theta_target)
        reconstruction_loss = nn.MSELoss()(reconstructed_theta, theta_target)
        latent_loss = nn.MSELoss()(evolved_latent, lifted_target)
        total_loss = (
            self.model_config.lambda_pred * prediction_loss
            + self.model_config.lambda_rec * reconstruction_loss
            + self.model_config.lambda_lat * latent_loss
        )
        return {
            "total_loss": total_loss,
            "prediction_loss": prediction_loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
        }

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        with torch.no_grad():
            context = move_tensor_to_device(context, self.device)
            batch_size = context.shape[0]
            noise_state = initial_noise if initial_noise is not None else self.sample_base_noise(batch_size)
            noise_state = move_tensor_to_device(noise_state, self.device)
            return self.forward(noise_state, context)

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "lifting_dim": self.model_config.lifting_dim,
                    "lambda_rec": self.model_config.lambda_rec,
                    "lambda_lat": self.model_config.lambda_lat,
                    "lambda_pred": self.model_config.lambda_pred,
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
    def load(cls, filepath: str, device: torch.device) -> "KoopmanFlow":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = KoopmanModelConfig(
            lifting_dim=model_config_dict["lifting_dim"],
            lambda_rec=model_config_dict["lambda_rec"],
            lambda_lat=model_config_dict["lambda_lat"],
            lambda_pred=model_config_dict["lambda_pred"],
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
