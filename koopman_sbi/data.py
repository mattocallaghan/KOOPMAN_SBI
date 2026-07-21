from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import sbibm
import torch
from torch.utils.data import Dataset

from koopman_sbi.config import ExperimentConfig
from koopman_sbi.paths import resolve_dataset_dir


def _safe_std(values: torch.Tensor) -> torch.Tensor:
    return torch.clamp(torch.std(values, dim=0), min=1e-6)


@dataclass
class Standardizer:
    theta_mean: torch.Tensor
    theta_std: torch.Tensor
    x_mean: torch.Tensor
    x_std: torch.Tensor

    @classmethod
    def from_training_tensors(cls, theta: torch.Tensor, x: torch.Tensor) -> "Standardizer":
        return cls(
            theta_mean=torch.mean(theta, dim=0),
            theta_std=_safe_std(theta),
            x_mean=torch.mean(x, dim=0),
            x_std=_safe_std(x),
        )

    def standardize_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return (theta - self.theta_mean.to(theta.device)) / self.theta_std.to(theta.device)

    def inverse_theta(self, theta: torch.Tensor) -> torch.Tensor:
        return theta * self.theta_std.to(theta.device) + self.theta_mean.to(theta.device)

    def standardize_x(self, x: torch.Tensor) -> torch.Tensor:
        return (x - self.x_mean.to(x.device)) / self.x_std.to(x.device)

    def inverse_x(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.x_std.to(x.device) + self.x_mean.to(x.device)


class SBIPairDataset(Dataset):
    def __init__(self, theta: torch.Tensor, x: torch.Tensor, standardizer: Standardizer):
        super().__init__()
        self.theta_raw = theta
        self.x_raw = x
        self.standardizer = standardizer
        self.theta = standardizer.standardize_theta(theta)
        self.x = standardizer.standardize_x(x)

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.theta[index], self.x[index]


@dataclass
class DatasetBundle:
    train_dataset: SBIPairDataset
    val_dataset: SBIPairDataset
    standardizer: Standardizer
    raw_theta: torch.Tensor
    raw_x: torch.Tensor
    dataset_dir: Path

    @property
    def dim_theta(self) -> int:
        return self.raw_theta.shape[1]

    @property
    def dim_x(self) -> int:
        return self.raw_x.shape[1]

    def standardized_context_pool(self) -> torch.Tensor:
        return torch.cat([self.train_dataset.x, self.val_dataset.x], dim=0)


def _simulate_dataset(config: ExperimentConfig) -> Tuple[torch.Tensor, torch.Tensor]:
    task = sbibm.get_task(config.task.name)
    prior = task.get_prior()
    simulator = task.get_simulator()
    num_samples = config.task.num_train_samples
    batch_size = config.task.simulation_batch_size
    num_batches = math.ceil(num_samples / batch_size)
    theta_batches = []
    x_batches = []
    for _ in range(num_batches):
        theta_batch = prior(batch_size)
        x_batch = simulator(theta_batch)
        theta_batches.append(theta_batch.detach().cpu())
        x_batches.append(x_batch.detach().cpu())
    theta = torch.cat(theta_batches, dim=0)[:num_samples].float()
    x = torch.cat(x_batches, dim=0)[:num_samples].float()
    return theta, x


def _save_raw_dataset(dataset_dir: Path, theta: torch.Tensor, x: torch.Tensor) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    np.save(dataset_dir / "theta.npy", theta.numpy())
    np.save(dataset_dir / "x.npy", x.numpy())


def _load_raw_dataset(dataset_dir: Path) -> Tuple[torch.Tensor, torch.Tensor]:
    theta = torch.tensor(np.load(dataset_dir / "theta.npy"), dtype=torch.float32)
    x = torch.tensor(np.load(dataset_dir / "x.npy"), dtype=torch.float32)
    return theta, x


def _split_indices(num_samples: int, train_fraction: float, seed: int) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(num_samples, generator=generator)
    num_train = int(num_samples * train_fraction)
    return permutation[:num_train], permutation[num_train:]


def load_or_generate_dataset(config: ExperimentConfig) -> DatasetBundle:
    dataset_dir = resolve_dataset_dir(config.logging.output_root, config.task.name, config.task.dataset_dir)
    theta_path = dataset_dir / "theta.npy"
    x_path = dataset_dir / "x.npy"
    if config.task.use_cached_dataset and theta_path.exists() and x_path.exists():
        raw_theta, raw_x = _load_raw_dataset(dataset_dir)
    else:
        raw_theta, raw_x = _simulate_dataset(config)
        _save_raw_dataset(dataset_dir, raw_theta, raw_x)

    train_indices, val_indices = _split_indices(len(raw_theta), config.task.train_fraction, config.task.seed)
    theta_train = raw_theta[train_indices]
    x_train = raw_x[train_indices]
    theta_val = raw_theta[val_indices]
    x_val = raw_x[val_indices]

    standardizer = Standardizer.from_training_tensors(theta_train, x_train)
    train_dataset = SBIPairDataset(theta_train, x_train, standardizer)
    val_dataset = SBIPairDataset(theta_val, x_val, standardizer)
    return DatasetBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        standardizer=standardizer,
        raw_theta=raw_theta,
        raw_x=raw_x,
        dataset_dir=dataset_dir,
    )
