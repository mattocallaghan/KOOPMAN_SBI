from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from koopman_sbi.config import ExperimentConfig
from koopman_sbi.data import DatasetBundle
from koopman_sbi.paths import resolve_teacher_dir
from koopman_sbi.runtime import move_tensor_to_device


_CACHE_METADATA_FILENAME = "metadata.json"
_TRAJECTORY_CACHE_VERSION = 1


class TeacherTrajectoryDataset(Dataset):
    def __init__(self, noise_state: torch.Tensor, theta: torch.Tensor, context: torch.Tensor):
        super().__init__()
        self.noise_state = noise_state
        self.theta = theta
        self.context = context

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.noise_state[index], self.theta[index], self.context[index]


@dataclass
class TeacherTrajectoryBundle:
    train_dataset: TeacherTrajectoryDataset
    val_dataset: TeacherTrajectoryDataset
    trajectory_dir: Path
    generation_time_seconds: float
    loaded_from_cache: bool
    num_samples: int
    num_context: int


def _load_trajectories(trajectory_dir: Path) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    noise_state = torch.tensor(np.load(trajectory_dir / "noise.npy"), dtype=torch.float32)
    theta = torch.tensor(np.load(trajectory_dir / "theta.npy"), dtype=torch.float32)
    context = torch.tensor(np.load(trajectory_dir / "context.npy"), dtype=torch.float32)
    return noise_state, theta, context


def _save_trajectories(
    trajectory_dir: Path,
    noise_state: torch.Tensor,
    theta: torch.Tensor,
    context: torch.Tensor,
) -> None:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.save(trajectory_dir / "noise.npy", noise_state.numpy())
    np.save(trajectory_dir / "theta.npy", theta.numpy())
    np.save(trajectory_dir / "context.npy", context.numpy())


def _hash_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _hash_tensor(tensor: torch.Tensor) -> str:
    values = tensor.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tuple(values.shape)).encode("utf-8"))
    digest.update(str(values.dtype).encode("utf-8"))
    digest.update(values.numpy().tobytes())
    return digest.hexdigest()


def _teacher_model_digest(teacher_model, checkpoint_path: Optional[Path]) -> str:
    if checkpoint_path is not None and checkpoint_path.exists():
        return _hash_file(checkpoint_path)
    digest = hashlib.sha256()
    for name, parameter in sorted(teacher_model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        digest.update(_hash_tensor(parameter).encode("utf-8"))
    return digest.hexdigest()


def _trajectory_cache_metadata(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    teacher_model,
    teacher_checkpoint_path: Optional[Path],
) -> Dict[str, Any]:
    return {
        "version": _TRAJECTORY_CACHE_VERSION,
        "task": {
            "name": config.task.name,
            "seed": config.task.seed,
            "num_train_samples": config.task.num_train_samples,
            "train_fraction": config.task.train_fraction,
        },
        "teacher": {
            "checkpoint_sha256": _teacher_model_digest(teacher_model, teacher_checkpoint_path),
            "num_samples": config.teacher.num_samples,
            "num_context": config.teacher.num_context,
            "batch_size": config.teacher.batch_size,
        },
        "dataset": {
            "theta_sha256": _hash_tensor(dataset_bundle.raw_theta),
            "x_sha256": _hash_tensor(dataset_bundle.raw_x),
        },
    }


def _cache_matches(trajectory_dir: Path, expected_metadata: Dict[str, Any]) -> bool:
    metadata_path = trajectory_dir / _CACHE_METADATA_FILENAME
    if not metadata_path.exists():
        return False
    try:
        with open(metadata_path, "r", encoding="utf-8") as handle:
            cached_metadata = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return cached_metadata == expected_metadata


def _save_cache_metadata(trajectory_dir: Path, metadata: Dict[str, Any]) -> None:
    with open(trajectory_dir / _CACHE_METADATA_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _split_teacher_data(
    noise_state: torch.Tensor,
    theta: torch.Tensor,
    context: torch.Tensor,
    train_fraction: float,
    seed: int,
) -> Tuple[TeacherTrajectoryDataset, TeacherTrajectoryDataset]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(theta), generator=generator)
    num_train = int(len(theta) * train_fraction)
    train_indices = permutation[:num_train]
    val_indices = permutation[num_train:]
    return (
        TeacherTrajectoryDataset(noise_state[train_indices], theta[train_indices], context[train_indices]),
        TeacherTrajectoryDataset(noise_state[val_indices], theta[val_indices], context[val_indices]),
    )


def _select_context_pool(dataset_bundle: DatasetBundle, num_context: int, seed: int) -> torch.Tensor:
    context_pool = dataset_bundle.standardized_context_pool()
    if num_context >= len(context_pool):
        return context_pool
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(context_pool), generator=generator)[:num_context]
    return context_pool[indices]


def load_or_generate_teacher_trajectories(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    teacher_model,
    device: torch.device,
    teacher_checkpoint_path: Optional[Path] = None,
) -> TeacherTrajectoryBundle:
    trajectory_dir = resolve_teacher_dir(
        config.logging.output_root,
        config.task.name,
        config.teacher.trajectory_dir,
    )
    noise_path = trajectory_dir / "noise.npy"
    theta_path = trajectory_dir / "theta.npy"
    context_path = trajectory_dir / "context.npy"
    cache_metadata = _trajectory_cache_metadata(
        config=config,
        dataset_bundle=dataset_bundle,
        teacher_model=teacher_model,
        teacher_checkpoint_path=teacher_checkpoint_path,
    )

    start_time = time.time()
    if (
        config.teacher.load_cached_trajectories
        and noise_path.exists()
        and theta_path.exists()
        and context_path.exists()
        and _cache_matches(trajectory_dir, cache_metadata)
    ):
        noise_state, theta, context = _load_trajectories(trajectory_dir)
        loaded_from_cache = True
    else:
        loaded_from_cache = False
        context_pool = _select_context_pool(dataset_bundle, config.teacher.num_context, config.task.seed)
        teacher_model.eval()
        noise_batches = []
        theta_batches = []
        context_batches = []
        total_samples = config.teacher.num_samples
        batch_size = config.teacher.batch_size
        with torch.no_grad():
            for start in range(0, total_samples, batch_size):
                current_batch = min(batch_size, total_samples - start)
                generator = torch.Generator().manual_seed(config.task.seed + start)
                indices = torch.randint(
                    low=0,
                    high=len(context_pool),
                    size=(current_batch,),
                    generator=generator,
                )
                context_batch = move_tensor_to_device(context_pool[indices], device)
                noise_generator = torch.Generator().manual_seed(
                    config.task.seed + total_samples + start
                )
                noise_batch = move_tensor_to_device(
                    torch.randn(
                        current_batch,
                        teacher_model.input_dim,
                        generator=noise_generator,
                        dtype=torch.float32,
                    ),
                    device,
                )
                theta_batch = teacher_model.sample_batch(context_batch, initial_noise=noise_batch)
                noise_batches.append(noise_batch.detach().cpu())
                theta_batches.append(theta_batch.detach().cpu())
                context_batches.append(context_batch.detach().cpu())
        noise_state = torch.cat(noise_batches, dim=0)
        theta = torch.cat(theta_batches, dim=0)
        context = torch.cat(context_batches, dim=0)
        if config.teacher.cache_trajectories:
            _save_trajectories(trajectory_dir, noise_state, theta, context)
            _save_cache_metadata(trajectory_dir, cache_metadata)
    generation_time_seconds = time.time() - start_time

    train_dataset, val_dataset = _split_teacher_data(
        noise_state=noise_state,
        theta=theta,
        context=context,
        train_fraction=config.task.train_fraction,
        seed=config.task.seed,
    )
    return TeacherTrajectoryBundle(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        trajectory_dir=trajectory_dir,
        generation_time_seconds=generation_time_seconds,
        loaded_from_cache=loaded_from_cache,
        num_samples=int(len(theta)),
        num_context=int(len(torch.unique(context, dim=0))) if len(context) > 0 else 0,
    )
