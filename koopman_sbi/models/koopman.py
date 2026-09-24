from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_sbi.config import AdversarialConfig, KoopmanModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.models.networks import DenseResidualNet
from koopman_sbi.runtime import move_tensor_to_device


class KoopmanDiscriminator(nn.Module):
    def __init__(
        self,
        theta_dim: int,
        context_dim: int,
        config: AdversarialConfig,
    ) -> None:
        super().__init__()
        self.network = DenseResidualNet(
            input_dim=theta_dim + context_dim,
            output_dim=1,
            hidden_dims=config.hidden_dims,
            activation=config.activation,
            batch_norm=config.batch_norm,
            dropout=config.dropout,
            theta_dim=theta_dim,
            context_dim=context_dim,
        )

    def forward(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.network(torch.cat([theta, context], dim=-1))


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
        self.discriminator = (
            KoopmanDiscriminator(theta_dim=input_dim, context_dim=context_dim, config=model_config.adversarial)
            if model_config.adversarial.enabled
            else None
        )

    def generator_parameters(self):
        for module in [self.encoder, self.koopman_linear, self.context_modulation, self.decoder]:
            yield from module.parameters()

    def discriminator_parameters(self):
        if self.discriminator is None:
            return
        yield from self.discriminator.parameters()

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

    def _compute_koopman_losses(
        self,
        noise_state: torch.Tensor,
        theta_target: torch.Tensor,
        context: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        lifted_noise = self.encode_noise(noise_state, context)
        evolved_latent = self.evolve_latent(lifted_noise, context)
        predicted_theta = self.decode_latent(evolved_latent, context)

        lifted_target = self.encode_target(theta_target, context)
        reconstructed_theta = self.decode_latent(lifted_target, context)

        prediction_loss = nn.MSELoss()(predicted_theta, theta_target)
        reconstruction_loss = nn.MSELoss()(reconstructed_theta, theta_target)
        latent_loss = nn.MSELoss()(evolved_latent, lifted_target)
        koopman_loss = (
            self.model_config.lambda_pred * prediction_loss
            + self.model_config.lambda_rec * reconstruction_loss
            + self.model_config.lambda_lat * latent_loss
        )
        return {
            "koopman_loss": koopman_loss,
            "prediction_loss": prediction_loss,
            "reconstruction_loss": reconstruction_loss,
            "latent_loss": latent_loss,
            "predicted_theta": predicted_theta,
        }

    def _generator_adversarial_loss(
        self,
        predicted_theta: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.discriminator is None:
            return predicted_theta.new_zeros(())
        fake_logits = self.discriminator(predicted_theta, context)
        return F.binary_cross_entropy_with_logits(fake_logits, torch.ones_like(fake_logits))

    def _discriminator_loss(
        self,
        theta_target: torch.Tensor,
        predicted_theta: torch.Tensor,
        context: torch.Tensor,
    ) -> torch.Tensor:
        if self.discriminator is None:
            return predicted_theta.new_zeros(())
        real_logits = self.discriminator(theta_target, context)
        fake_logits = self.discriminator(predicted_theta.detach(), context)
        real_loss = F.binary_cross_entropy_with_logits(real_logits, torch.ones_like(real_logits))
        fake_loss = F.binary_cross_entropy_with_logits(fake_logits, torch.zeros_like(fake_logits))
        return 0.5 * (real_loss + fake_loss)

    def compute_loss(self, batch: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del kwargs
        noise_state, theta_target, context = batch
        koopman_losses = self._compute_koopman_losses(noise_state, theta_target, context)
        adversarial_loss = self._generator_adversarial_loss(koopman_losses["predicted_theta"], context)
        total_loss = koopman_losses["koopman_loss"] + self.model_config.adversarial.lambda_adv * adversarial_loss
        metrics = {
            "total_loss": total_loss,
            "koopman_loss": koopman_losses["koopman_loss"],
            "prediction_loss": koopman_losses["prediction_loss"],
            "reconstruction_loss": koopman_losses["reconstruction_loss"],
            "latent_loss": koopman_losses["latent_loss"],
        }
        if self.discriminator is not None:
            metrics["generator_adversarial_loss"] = adversarial_loss
            metrics["discriminator_loss"] = self._discriminator_loss(
                theta_target,
                koopman_losses["predicted_theta"],
                context,
            )
        return metrics

    def train_batch(
        self,
        batch: Any,
        optimizer: torch.optim.Optimizer,
        discriminator_optimizer: Optional[torch.optim.Optimizer] = None,
        gradient_clip_norm: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        noise_state, theta_target, context = batch
        koopman_losses = self._compute_koopman_losses(noise_state, theta_target, context)
        predicted_theta = koopman_losses["predicted_theta"]

        discriminator_loss = predicted_theta.new_zeros(())
        if self.discriminator is not None and discriminator_optimizer is not None:
            discriminator_optimizer.zero_grad()
            discriminator_loss = self._discriminator_loss(theta_target, predicted_theta, context)
            discriminator_loss.backward()
            discriminator_optimizer.step()

        optimizer.zero_grad()
        adversarial_loss = self._generator_adversarial_loss(predicted_theta, context)
        total_loss = koopman_losses["koopman_loss"] + self.model_config.adversarial.lambda_adv * adversarial_loss
        total_loss.backward()
        if gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(list(self.generator_parameters()), gradient_clip_norm)
        optimizer.step()

        metrics = {
            "total_loss": total_loss.detach(),
            "koopman_loss": koopman_losses["koopman_loss"].detach(),
            "prediction_loss": koopman_losses["prediction_loss"].detach(),
            "reconstruction_loss": koopman_losses["reconstruction_loss"].detach(),
            "latent_loss": koopman_losses["latent_loss"].detach(),
        }
        if self.discriminator is not None:
            metrics["generator_adversarial_loss"] = adversarial_loss.detach()
            metrics["discriminator_loss"] = discriminator_loss.detach()
        return metrics

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
                    "adversarial": {
                        "enabled": self.model_config.adversarial.enabled,
                        "lambda_adv": self.model_config.adversarial.lambda_adv,
                        "hidden_dims": self.model_config.adversarial.hidden_dims,
                        "activation": self.model_config.adversarial.activation,
                        "batch_norm": self.model_config.adversarial.batch_norm,
                        "dropout": self.model_config.adversarial.dropout,
                    },
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
            adversarial=AdversarialConfig(**model_config_dict.get("adversarial", {})),
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
