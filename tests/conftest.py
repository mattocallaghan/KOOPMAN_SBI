from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import yaml


if "sbibm" not in sys.modules:
    sbibm_stub = types.ModuleType("sbibm")
    sbibm_stub.get_task = lambda name: None
    sbibm_metrics_stub = types.ModuleType("sbibm.metrics")
    for metric_name in [
        "c2st",
        "ksd",
        "median_distance",
        "mmd",
        "posterior_mean_error",
        "posterior_variance_ratio",
    ]:
        setattr(sbibm_metrics_stub, metric_name, lambda *args, **kwargs: torch.tensor(0.0))
    sbibm_tasks_stub = types.ModuleType("sbibm.tasks")
    sbibm_stub.metrics = sbibm_metrics_stub
    sbibm_stub.tasks = sbibm_tasks_stub
    sys.modules["sbibm"] = sbibm_stub
    sys.modules["sbibm.metrics"] = sbibm_metrics_stub
    sys.modules["sbibm.tasks"] = sbibm_tasks_stub

if "torchdiffeq" not in sys.modules:
    torchdiffeq_stub = types.ModuleType("torchdiffeq")

    def _odeint(func, y0, t, atol=None, rtol=None, method=None, options=None):
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

    torchdiffeq_stub.odeint = _odeint
    sys.modules["torchdiffeq"] = torchdiffeq_stub

if "normflows" not in sys.modules:
    normflows_stub = types.ModuleType("normflows")

    class _DiagGaussian:
        def __init__(self, dim, trainable=False):
            self.dim = dim
            self.trainable = trainable

    class _MaskedAffineAutoregressive(torch.nn.Module):
        def __init__(
            self,
            features,
            hidden_features,
            context_features,
            num_blocks,
            use_residual_blocks,
            random_mask,
            activation,
            dropout_probability,
            use_batch_norm,
        ):
            super().__init__()
            del hidden_features, num_blocks, use_residual_blocks, random_mask, activation, dropout_probability, use_batch_norm
            self.linear = torch.nn.Linear(features + context_features, features)

        def forward(self, theta, context):
            return self.linear(torch.cat([theta, context], dim=-1))

    class _Permute(torch.nn.Module):
        def __init__(self, features, mode="swap"):
            super().__init__()
            self.features = features
            self.mode = mode

        def forward(self, theta, context):
            del context
            if self.mode == "swap" and theta.shape[-1] > 1:
                return torch.flip(theta, dims=[-1])
            return theta

    class _ConditionalNormalizingFlow(torch.nn.Module):
        def __init__(self, q0, flows):
            super().__init__()
            self.q0 = q0
            self.flows = torch.nn.ModuleList(flows)
            self.context_encoder = torch.nn.LazyLinear(q0.dim)

        def log_prob(self, theta, context=None):
            mean = self.context_encoder(context)
            return -torch.sum((theta - mean) ** 2, dim=-1)

        def forward_kld(self, theta, context=None):
            return -self.log_prob(theta, context=context).mean()

        def sample(self, num_samples, context=None):
            context = context.expand(num_samples, -1)
            mean = self.context_encoder(context)
            sample = mean + 0.05 * torch.randn_like(mean)
            return sample.unsqueeze(0), torch.zeros(num_samples, device=sample.device)

    distributions = types.SimpleNamespace(base=types.SimpleNamespace(DiagGaussian=_DiagGaussian))
    flows = types.SimpleNamespace(
        MaskedAffineAutoregressive=_MaskedAffineAutoregressive,
        Permute=_Permute,
    )
    normflows_stub.distributions = distributions
    normflows_stub.flows = flows
    normflows_stub.ConditionalNormalizingFlow = _ConditionalNormalizingFlow
    sys.modules["normflows"] = normflows_stub

from koopman_sbi.evaluation import SAMPLE_METRICS


class FakeTask:
    def __init__(self, dim_theta: int = 2, dim_x: int = 2) -> None:
        self.dim_theta = dim_theta
        self.dim_x = dim_x
        self.prior_dist = torch.distributions.Independent(
            torch.distributions.Normal(torch.zeros(dim_theta), 2.0 * torch.ones(dim_theta)),
            1,
        )

    def get_prior(self):
        def prior(batch_size: int) -> torch.Tensor:
            return torch.randn(batch_size, self.dim_theta)

        return prior

    def get_simulator(self):
        def simulator(theta: torch.Tensor) -> torch.Tensor:
            return theta[:, : self.dim_x] + 0.05 * torch.randn(theta.shape[0], self.dim_x)

        return simulator

    def get_observation(self, num_observation: int) -> torch.Tensor:
        generator = torch.Generator().manual_seed(100 + num_observation)
        return torch.randn(self.dim_x, generator=generator)

    def get_reference_posterior_samples(self, num_observation: int) -> torch.Tensor:
        observation = self.get_observation(num_observation)
        generator = torch.Generator().manual_seed(200 + num_observation)
        return observation.repeat(128, 1) + 0.1 * torch.randn(128, self.dim_theta, generator=generator)


@pytest.fixture(autouse=True)
def fake_sbibm(monkeypatch):
    import sbibm

    monkeypatch.setattr(sbibm, "get_task", lambda name: FakeTask())
    patched_metrics = {
        "c2st": lambda task, obs, posterior, reference: float(torch.mean((posterior - reference) ** 2).item()),
        "mmd": lambda task, obs, posterior, reference: float(torch.mean(torch.abs(posterior - reference)).item()),
        "posterior_mean_error": lambda task, obs, posterior, reference: float(
            torch.mean(torch.abs(torch.mean(posterior, dim=0) - torch.mean(reference, dim=0))).item()
        ),
        "posterior_variance_ratio": lambda task, obs, posterior, reference: float(
            torch.mean(torch.var(posterior, dim=0) / (torch.var(reference, dim=0) + 1e-6)).item()
        ),
        "median_distance": lambda task, obs, posterior, reference: float(
            torch.median(torch.norm(posterior - reference, dim=-1)).item()
        ),
        "ksd": lambda task, obs, posterior, reference: float(torch.mean(posterior).item()),
    }
    monkeypatch.setattr("koopman_sbi.evaluation.SAMPLE_METRICS", patched_metrics)
    return sbibm


@pytest.fixture
def tiny_config_path(tmp_path: Path) -> Path:
    config = {
        "task": {
            "name": "two_moons",
            "seed": 0,
            "num_train_samples": 64,
            "simulation_batch_size": 16,
            "train_fraction": 0.75,
            "use_cached_dataset": False,
            "dataset_dir": str(tmp_path / "dataset"),
        },
        "model": {
            "flow_matching": {
                "sigma_min": 1e-4,
                "time_prior_exponent": 2.0,
                "atol": 1e-5,
                "rtol": 1e-5,
                "network": {
                    "type": "DenseResidualNet",
                    "hidden_dims": [16, 16],
                    "activation": "gelu",
                    "batch_norm": False,
                    "dropout": 0.0,
                    "theta_with_glu": False,
                    "context_with_glu": False,
                },
            },
            "koopman": {
                "lifting_dim": 12,
                "lambda_rec": 1.0,
                "lambda_lat": 1.0,
                "lambda_pred": 1.0,
                "network": {
                    "type": "DenseResidualNet",
                    "hidden_dims": [16, 16],
                    "activation": "gelu",
                    "batch_norm": False,
                    "dropout": 0.0,
                    "theta_with_glu": False,
                    "context_with_glu": False,
                },
            },
            "cmpe": {
                "eps": 1e-3,
                "t_max": 200.0,
                "rho": 7.0,
                "sigma_data": 1.0,
                "s0": 10,
                "s1": 50,
                "p_mean": -1.1,
                "p_std": 2.0,
                "default_num_steps": 10,
                "network": {
                    "type": "DenseResidualNet",
                    "hidden_dims": [16, 16],
                    "activation": "gelu",
                    "batch_norm": False,
                    "dropout": 0.0,
                    "theta_with_glu": False,
                    "context_with_glu": False,
                },
            },
        },
        "teacher": {
            "checkpoint_path": None,
            "auto_train_if_missing": True,
            "num_samples": 32,
            "num_context": 16,
            "batch_size": 8,
            "cache_trajectories": False,
            "load_cached_trajectories": False,
            "trajectory_dir": str(tmp_path / "teacher"),
        },
        "training": {
            "flow_matching": {
                "batch_size": 8,
                "epochs": 1,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": True,
                "patience": 2,
                "gradient_clip_norm": None,
                "precision": "float32",
                "use_tensorboard": False,
                "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0},
                "scheduler": {"type": "StepLR", "factor": 0.5, "patience": 1, "step_size": 1, "gamma": 0.9},
            },
            "koopman": {
                "batch_size": 8,
                "epochs": 1,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": True,
                "patience": 2,
                "gradient_clip_norm": None,
                "precision": "float32",
                "use_tensorboard": False,
                "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0},
                "scheduler": {"type": "StepLR", "factor": 0.5, "patience": 1, "step_size": 1, "gamma": 0.9},
            },
            "cmpe": {
                "batch_size": 8,
                "epochs": 1,
                "num_workers": 0,
                "device": "cpu",
                "early_stopping": True,
                "patience": 2,
                "gradient_clip_norm": None,
                "precision": "float32",
                "use_tensorboard": False,
                "optimizer": {"name": "Adam", "lr": 1e-3, "weight_decay": 0.0},
                "scheduler": {"type": "StepLR", "factor": 0.5, "patience": 1, "step_size": 1, "gamma": 0.9},
            },
        },
        "evaluation": {
            "num_posterior_samples": 32,
            "observations": [1, 2],
            "metrics": ["c2st", "mmd", "posterior_mean_error", "posterior_variance_ratio"],
            "flow_checkpoint_path": None,
            "koopman_checkpoint_path": None,
            "npe_checkpoint_path": None,
            "cmpe_checkpoint_path": None,
            "save_observation_plots": False,
        },
        "logging": {
            "output_root": str(tmp_path / "logs"),
            "run_name": "test_run",
            "use_tensorboard": False,
            "use_wandb": False,
            "wandb_project": "koopman-sbi-tests",
            "wandb_run_name": None,
            "wandb_tags": [],
        },
        "benchmark_suite": {
            "variants": [
                {"name": "npe", "model_type": "npe", "sample_kwargs": {}},
                {
                    "name": "fmnpe_dopri5",
                    "model_type": "flow_matching",
                    "sample_kwargs": {"solver": "dopri5", "atol": 1e-5, "rtol": 1e-5},
                },
                {
                    "name": "fmnpe_rk4_10",
                    "model_type": "flow_matching",
                    "sample_kwargs": {"solver": "rk4", "integration_steps": 10},
                },
                {"name": "cmpe_10", "model_type": "cmpe", "sample_kwargs": {"num_steps": 10}},
                {"name": "koopman", "model_type": "koopman", "sample_kwargs": {}},
            ]
        },
    }
    config_path = tmp_path / "config.yaml"
    with open(config_path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config, handle, sort_keys=False)
    return config_path
