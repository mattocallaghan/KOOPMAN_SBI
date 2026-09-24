from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn

from koopman_sbi.config import NetworkConfig, TensorProductKoopmanModelConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


def _network_config_to_dict(config: NetworkConfig) -> Dict[str, object]:
    return {
        "hidden_dims": config.hidden_dims,
        "activation": config.activation,
        "batch_norm": config.batch_norm,
        "dropout": config.dropout,
        "theta_with_glu": config.theta_with_glu,
        "context_with_glu": config.context_with_glu,
        "type": config.type,
    }


class TensorProductKoopmanFlow(BasePosteriorModel):
    """One-shot Koopman distillation model with a low-rank bilinear context operator."""

    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: TensorProductKoopmanModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        if int(model_config.latent_dim) < 1:
            raise ValueError("tensorproduct_koopman.latent_dim must be positive.")
        if int(model_config.context_feature_dim) < 1:
            raise ValueError("tensorproduct_koopman.context_feature_dim must be positive.")
        if int(model_config.tensor_rank) < 1:
            raise ValueError("tensorproduct_koopman.tensor_rank must be positive.")

        state_cfg = model_config.state_network
        context_cfg = model_config.context_network
        decoder_cfg = model_config.decoder_network
        latent_dim = int(model_config.latent_dim)
        context_feature_dim = int(model_config.context_feature_dim)
        tensor_rank = int(model_config.tensor_rank)

        self.state_encoder = DenseResidualNet(
            input_dim=input_dim,
            output_dim=latent_dim,
            hidden_dims=state_cfg.hidden_dims,
            activation=state_cfg.activation,
            batch_norm=state_cfg.batch_norm,
            dropout=state_cfg.dropout,
            theta_dim=input_dim,
            theta_with_glu=state_cfg.theta_with_glu,
        )
        self.context_encoder = DenseResidualNet(
            input_dim=context_dim,
            output_dim=context_feature_dim,
            hidden_dims=context_cfg.hidden_dims,
            activation=context_cfg.activation,
            batch_norm=context_cfg.batch_norm,
            dropout=context_cfg.dropout,
            context_dim=context_dim,
            context_with_glu=context_cfg.context_with_glu,
        )
        self.decoder = DenseResidualNet(
            input_dim=latent_dim,
            output_dim=input_dim,
            hidden_dims=decoder_cfg.hidden_dims,
            activation=decoder_cfg.activation,
            batch_norm=decoder_cfg.batch_norm,
            dropout=decoder_cfg.dropout,
            theta_dim=latent_dim,
            theta_with_glu=decoder_cfg.theta_with_glu,
        )
        self.base_operator = nn.Linear(latent_dim, latent_dim, bias=False)
        self.state_factor = nn.Linear(latent_dim, tensor_rank, bias=False)
        self.context_factor = nn.Linear(context_feature_dim, tensor_rank, bias=False)
        self.factor_reconstruction = nn.Linear(tensor_rank, latent_dim, bias=False)

    def encode_state(self, state: torch.Tensor) -> torch.Tensor:
        return self.state_encoder(state)

    def encode_context(self, context: torch.Tensor) -> torch.Tensor:
        return self.context_encoder(context)

    def decode_state(self, latent: torch.Tensor) -> torch.Tensor:
        return self.decoder(latent)

    def evolve_latent(self, latent_state: torch.Tensor, context_feature: torch.Tensor) -> torch.Tensor:
        state_factor = self.state_factor(latent_state)
        context_factor = self.context_factor(context_feature)
        bilinear_update = self.factor_reconstruction(state_factor * context_factor)
        return self.base_operator(latent_state) + bilinear_update

    def forward(self, initial_state: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        latent_initial = self.encode_state(initial_state)
        context_feature = self.encode_context(context)
        latent_predicted = self.evolve_latent(latent_initial, context_feature)
        return self.decode_state(latent_predicted)

    def compute_loss(self, batch: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del kwargs
        initial_state, target_state, context = batch
        latent_initial = self.encode_state(initial_state)
        latent_target = self.encode_state(target_state)
        context_feature = self.encode_context(context)
        latent_predicted = self.evolve_latent(latent_initial, context_feature)

        reconstructed_initial = self.decode_state(latent_initial)
        reconstructed_target = self.decode_state(latent_target)
        predicted_target = self.decode_state(latent_predicted)

        ae_loss = nn.MSELoss()(reconstructed_initial, initial_state) + nn.MSELoss()(
            reconstructed_target,
            target_state,
        )
        latent_loss = nn.MSELoss()(latent_predicted, latent_target)
        endpoint_loss = nn.MSELoss()(predicted_target, target_state)
        total_loss = (
            float(self.model_config.lambda_ae) * ae_loss
            + float(self.model_config.lambda_lat) * latent_loss
            + float(self.model_config.lambda_end) * endpoint_loss
        )
        return {
            "total_loss": total_loss,
            "autoencoder_loss": ae_loss,
            "latent_loss": latent_loss,
            "endpoint_loss": endpoint_loss,
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
            initial_state = (
                initial_noise
                if initial_noise is not None
                else torch.randn(batch_size, self.input_dim, device=self.device)
            )
            initial_state = move_tensor_to_device(initial_state, self.device)
            return self.forward(initial_state, context)

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "latent_dim": self.model_config.latent_dim,
                    "context_feature_dim": self.model_config.context_feature_dim,
                    "tensor_rank": self.model_config.tensor_rank,
                    "lambda_ae": self.model_config.lambda_ae,
                    "lambda_lat": self.model_config.lambda_lat,
                    "lambda_end": self.model_config.lambda_end,
                    "state_network": _network_config_to_dict(self.model_config.state_network),
                    "context_network": _network_config_to_dict(self.model_config.context_network),
                    "decoder_network": _network_config_to_dict(self.model_config.decoder_network),
                },
            },
            filepath,
        )

    @classmethod
    def load(cls, filepath: str, device: torch.device) -> "TensorProductKoopmanFlow":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = TensorProductKoopmanModelConfig(
            latent_dim=model_config_dict.get("latent_dim", 256),
            context_feature_dim=model_config_dict.get("context_feature_dim", 128),
            tensor_rank=model_config_dict.get("tensor_rank", 64),
            lambda_ae=model_config_dict.get("lambda_ae", 1.0),
            lambda_lat=model_config_dict.get("lambda_lat", 1.0),
            lambda_end=model_config_dict.get("lambda_end", 1.0),
            state_network=NetworkConfig(**model_config_dict["state_network"]),
            context_network=NetworkConfig(**model_config_dict["context_network"]),
            decoder_network=NetworkConfig(**model_config_dict["decoder_network"]),
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
