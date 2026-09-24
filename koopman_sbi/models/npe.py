from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_sbi.config import NPEModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel
from koopman_sbi.runtime import move_tensor_to_device


def _activation_from_name(name: str):
    if name == "gelu":
        return F.gelu
    if name == "relu":
        return F.relu
    if name == "elu":
        return F.elu
    if name == "tanh":
        return torch.tanh
    raise ValueError(f"Unsupported activation for NPE MAF: {name}")


def _activation_module_from_name(name: str):
    if name == "gelu":
        return nn.GELU
    if name == "relu":
        return nn.ReLU
    if name == "elu":
        return nn.ELU
    if name == "tanh":
        return nn.Tanh
    raise ValueError(f"Unsupported activation for NPE NSF: {name}")


class NormalizingFlowNPE(BasePosteriorModel):
    def __init__(
        self,
        input_dim: int,
        context_dim: int,
        model_config: NPEModelConfig,
        device: torch.device,
    ) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        self.flow = self._build_flow()

    def _import_normflows(self):
        try:
            import normflows as nf
        except ImportError as exc:
            raise ImportError(
                "NPE requires the `normflows` package in the active environment."
            ) from exc
        return nf

    def _build_flow(self):
        if self.model_config.transform not in {"affine", "neural_spline"}:
            raise ValueError(
                f"Unsupported NPE transform '{self.model_config.transform}'. "
                "Use 'affine' or 'neural_spline'."
            )
        nf = self._import_normflows()
        network_cfg = self.model_config.network
        hidden_features = int(max(network_cfg.hidden_dims))
        num_blocks = int(len(network_cfg.hidden_dims))
        activation = _activation_from_name(network_cfg.activation)

        base_distribution = self.model_config.base_distribution.lower()
        if base_distribution != "normal":
            raise ValueError(
                f"Unsupported NPE base distribution '{self.model_config.base_distribution}'. "
                "Use 'normal'."
            )
        q0 = nf.distributions.base.DiagGaussian(self.input_dim, trainable=False)

        flows = []
        for layer_index in range(int(self.model_config.num_coupling_layers)):
            if self.model_config.transform == "affine":
                flows.append(
                    nf.flows.MaskedAffineAutoregressive(
                        features=self.input_dim,
                        hidden_features=hidden_features,
                        context_features=self.context_dim,
                        num_blocks=num_blocks,
                        use_residual_blocks=True,
                        random_mask=False,
                        activation=activation,
                        dropout_probability=float(network_cfg.dropout),
                        use_batch_norm=bool(network_cfg.batch_norm),
                    )
                )
            else:
                flows.append(
                    nf.flows.AutoregressiveRationalQuadraticSpline(
                        num_input_channels=self.input_dim,
                        num_blocks=num_blocks,
                        num_hidden_channels=hidden_features,
                        num_context_channels=self.context_dim,
                        num_bins=int(self.model_config.spline_num_bins),
                        tail_bound=float(self.model_config.spline_tail_bound),
                        activation=_activation_module_from_name(network_cfg.activation),
                        dropout_probability=float(network_cfg.dropout),
                        permute_mask=bool(layer_index % 2),
                        init_identity=bool(self.model_config.spline_init_identity),
                    )
                )
            if layer_index < int(self.model_config.num_coupling_layers) - 1 and self.model_config.permutation:
                flows.append(nf.flows.Permute(self.input_dim, mode=str(self.model_config.permutation)))

        flow = nf.ConditionalNormalizingFlow(q0, flows)
        return flow.to(self.device)

    def forward(self, theta: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        return self.flow.log_prob(theta, context=context)

    def compute_loss(self, batch: Any, **kwargs: Any) -> Dict[str, torch.Tensor]:
        del kwargs
        theta_target, context = batch
        negative_log_likelihood = self.flow.forward_kld(theta_target, context=context)
        return {
            "total_loss": negative_log_likelihood,
            "negative_log_likelihood": negative_log_likelihood,
        }

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        self.eval()
        context = move_tensor_to_device(context, self.device)
        with torch.no_grad():
            if initial_noise is None:
                # ConditionalNormalizingFlow.sample cannot batch distinct contexts: its
                # base distribution ignores context and produces a single sample. Draw
                # the base batch directly, then apply the same forward flow transforms.
                if not hasattr(self.flow.q0, "forward"):
                    # Supports the lightweight normflows test double while the real
                    # implementation follows the batched path below.
                    samples, _ = self.flow.sample(num_samples=context.shape[0], context=context)
                    return samples[0] if samples.dim() == 3 else samples
                samples, _ = self.flow.q0.forward(num_samples=context.shape[0])
            else:
                samples = move_tensor_to_device(initial_noise, self.device)
                if samples.shape != (context.shape[0], self.input_dim):
                    raise ValueError(
                        "initial_noise must have shape "
                        f"({context.shape[0]}, {self.input_dim}), got {tuple(samples.shape)}."
                    )
            for flow in self.flow.flows:
                samples, _ = flow(samples, context=context)
        return samples

    def save(self, filepath: str) -> None:
        torch.save(
            {
                "model_state_dict": self.state_dict(),
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "model_config": {
                    "backend": self.model_config.backend,
                    "num_coupling_layers": self.model_config.num_coupling_layers,
                    "transform": self.model_config.transform,
                    "permutation": self.model_config.permutation,
                    "use_actnorm": self.model_config.use_actnorm,
                    "base_distribution": self.model_config.base_distribution,
                    "spline_num_bins": self.model_config.spline_num_bins,
                    "spline_tail_bound": self.model_config.spline_tail_bound,
                    "spline_init_identity": self.model_config.spline_init_identity,
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
    def load(cls, filepath: str, device: torch.device) -> "NormalizingFlowNPE":
        checkpoint = torch.load(filepath, map_location=device, weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = NPEModelConfig(
            backend=model_config_dict["backend"],
            num_coupling_layers=model_config_dict["num_coupling_layers"],
            transform=model_config_dict["transform"],
            permutation=model_config_dict["permutation"],
            use_actnorm=model_config_dict["use_actnorm"],
            base_distribution=model_config_dict["base_distribution"],
            spline_num_bins=model_config_dict.get("spline_num_bins", 8),
            spline_tail_bound=model_config_dict.get("spline_tail_bound", 3.0),
            spline_init_identity=model_config_dict.get("spline_init_identity", True),
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
