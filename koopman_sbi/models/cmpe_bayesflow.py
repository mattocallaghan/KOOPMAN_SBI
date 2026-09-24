from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from koopman_sbi.config import CMPEModelConfig, NetworkConfig
from koopman_sbi.models.base import BasePosteriorModel


def _prepare_bayesflow_environment() -> None:
    mpl_config_dir = Path.cwd() / ".cache" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ["KERAS_BACKEND"] = "torch"


def _build_bayesflow_approximator(
    input_dim: int,
    context_dim: int,
    model_config: CMPEModelConfig,
):
    _prepare_bayesflow_environment()
    import bayesflow as bf
    import keras

    total_steps = max(int(model_config.s0), int(model_config.s1), 2)
    consistency_model = bf.networks.ConsistencyModel(
        total_steps=total_steps,
        subnet_kwargs={
            "dropout": float(model_config.network.dropout),
            "widths": tuple(int(width) for width in model_config.network.hidden_dims),
            "activation": model_config.network.activation,
        },
        max_time=float(model_config.t_max),
        sigma2=float(model_config.sigma_data),
        eps=float(model_config.eps),
        s0=int(model_config.s0),
        s1=int(model_config.s1),
    )
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )
    workflow = bf.BasicWorkflow(
        simulator=None,
        adapter=adapter,
        inference_network=consistency_model,
        initial_learning_rate=1e-4,
        optimizer=keras.optimizers.AdamW,
    )
    build_data = {
        "theta": np.zeros((1, input_dim), dtype=np.float32),
        "x": np.zeros((1, context_dim), dtype=np.float32),
    }
    workflow.approximator.build_from_data(workflow.approximator.adapter(build_data, batched=True))
    workflow.approximator.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=1e-4, weight_decay=0.0),
        run_eagerly=True,
    )
    return workflow.approximator


class BayesFlowConsistencyModel(BasePosteriorModel):
    def __init__(
        self,
        approximator: Any,
        input_dim: int,
        context_dim: int,
        model_config: CMPEModelConfig,
        device: torch.device,
        theta_mean: torch.Tensor,
        theta_std: torch.Tensor,
        x_mean: torch.Tensor,
        x_std: torch.Tensor,
    ) -> None:
        super().__init__()
        self.approximator = approximator
        self.input_dim = input_dim
        self.context_dim = context_dim
        self.model_config = model_config
        self.device = device
        self.theta_mean = theta_mean.detach().cpu()
        self.theta_std = theta_std.detach().cpu()
        self.x_mean = x_mean.detach().cpu()
        self.x_std = x_std.detach().cpu()

    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError("BayesFlowConsistencyModel is trained through BayesFlow workflows, not Trainer.")

    def sample_batch(
        self,
        context: torch.Tensor,
        initial_noise: Optional[torch.Tensor] = None,
        num_steps: Optional[int] = None,
    ) -> torch.Tensor:
        del initial_noise
        context_cpu = context.detach().cpu()
        raw_x = context_cpu * self.x_std + self.x_mean
        kwargs = {"conditions": {"x": raw_x.numpy().astype(np.float32)}, "num_samples": 1}
        if num_steps is not None:
            kwargs["n_steps"] = int(num_steps)
        samples = self.approximator.sample(**kwargs)
        if isinstance(samples, dict):
            theta = samples["theta"]
        else:
            theta = samples
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim == 3 and theta.shape[1] == 1:
            theta = theta[:, 0, :]
        theta_tensor = torch.tensor(theta, dtype=torch.float32)
        standardized = (theta_tensor - self.theta_mean) / self.theta_std
        return standardized.to(self.device)

    def save(self, filepath: str) -> None:
        _prepare_bayesflow_environment()
        import bayesflow  # noqa: F401

        keras_path = Path(filepath).with_suffix(".keras")
        weights_path = Path(filepath).with_suffix(".weights.h5")
        self.approximator.save(keras_path)
        self.approximator.save_weights(weights_path)
        torch.save(
            {
                "backend": "bayesflow",
                "keras_path": keras_path.name,
                "weights_path": weights_path.name,
                "input_dim": self.input_dim,
                "context_dim": self.context_dim,
                "theta_mean": self.theta_mean,
                "theta_std": self.theta_std,
                "x_mean": self.x_mean,
                "x_std": self.x_std,
                "model_config": {
                    "backend": self.model_config.backend,
                    "eps": self.model_config.eps,
                    "t_max": self.model_config.t_max,
                    "rho": self.model_config.rho,
                    "sigma_data": self.model_config.sigma_data,
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
    def load(cls, filepath: str, device: torch.device) -> "BayesFlowConsistencyModel":
        _prepare_bayesflow_environment()
        import bayesflow  # noqa: F401
        import keras

        checkpoint = torch.load(filepath, map_location="cpu", weights_only=False)
        model_config_dict = checkpoint["model_config"]
        model_config = CMPEModelConfig(
            backend=model_config_dict.get("backend", "bayesflow"),
            eps=model_config_dict.get("eps", 1e-3),
            t_max=model_config_dict.get("t_max", 80.0),
            rho=model_config_dict.get("rho", 7.0),
            sigma_data=model_config_dict.get("sigma_data", 1.0),
            s0=model_config_dict.get("s0", 10),
            s1=model_config_dict.get("s1", 150),
            p_mean=model_config_dict.get("p_mean", -1.1),
            p_std=model_config_dict.get("p_std", 2.0),
            default_num_steps=model_config_dict.get("default_num_steps", 10),
            network=NetworkConfig(**model_config_dict["network"]),
        )
        keras_path = Path(filepath).with_name(checkpoint["keras_path"])
        weights_path = Path(filepath).with_name(checkpoint.get("weights_path", Path(filepath).with_suffix(".weights.h5").name))
        try:
            approximator = keras.saving.load_model(keras_path)
        except Exception:
            approximator = _build_bayesflow_approximator(
                input_dim=int(checkpoint["input_dim"]),
                context_dim=int(checkpoint["context_dim"]),
                model_config=model_config,
            )
            if not weights_path.exists():
                raise
            approximator.load_weights(weights_path)
        return cls(
            approximator=approximator,
            input_dim=int(checkpoint["input_dim"]),
            context_dim=int(checkpoint["context_dim"]),
            model_config=model_config,
            device=device,
            theta_mean=checkpoint["theta_mean"],
            theta_std=checkpoint["theta_std"],
            x_mean=checkpoint["x_mean"],
            x_std=checkpoint["x_std"],
        )
