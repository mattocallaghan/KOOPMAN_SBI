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
    def __init__(
        self,
        noise_state: torch.Tensor,
        theta: torch.Tensor,
        context: torch.Tensor,
        time_grid: Optional[torch.Tensor] = None,
        path: Optional[torch.Tensor] = None,
        path_mask: Optional[torch.Tensor] = None,
        include_full_trajectory: bool = False,
    ):
        super().__init__()
        self.noise_state = noise_state
        self.theta = theta
        self.context = context
        self.time_grid = time_grid
        self.path = path
        self.path_mask = path_mask
        self.include_full_trajectory = include_full_trajectory
        # Optional per-pair endpoint-loss metric (see attach_pullback_endpoint_metric): a tensor, or for
        # "vjp_sketch" a dict {"low_rank_factors": (n, d, k)}.
        self.endpoint_metric: Optional[torch.Tensor | Dict[str, torch.Tensor]] = None
        # Rows of the full trajectory cache held by this split (set by _split_teacher_data).
        self.source_indices: Optional[torch.Tensor] = None

    def __len__(self) -> int:
        return len(self.theta)

    def __getitem__(self, index: int):
        if (
            self.include_full_trajectory
            and self.time_grid is not None
            and self.path is not None
            and self.path_mask is not None
        ):
            return (
                self.noise_state[index],
                self.theta[index],
                self.context[index],
                self.time_grid[index],
                self.path[index],
                self.path_mask[index],
            )
        if self.endpoint_metric is not None:
            metric = self.endpoint_metric
            item = {key: value[index] for key, value in metric.items()} if isinstance(metric, dict) else metric[index]
            return self.noise_state[index], self.theta[index], self.context[index], item
        return self.noise_state[index], self.theta[index], self.context[index]


_JACOBIAN_METADATA_FILENAME = "endpoint_jacobian_metadata.json"
JACOBIAN_TYPES = ("full", "diagonal", "trace", "finite_difference_trace", "vjp_sketch")
# Types computed in the same solve as the trajectories; "vjp_sketch" runs as its own pass at its own tolerance.
_SINGLE_SOLVE_TYPES = ("full", "diagonal", "trace", "finite_difference_trace")
# Forward-difference step for "finite_difference_trace" (theta is standardised, so O(1) in scale).
_FINITE_DIFFERENCE_STEP = 1e-3


def _teacher_flow_sensitivity(
    teacher_model,
    noise: torch.Tensor,
    context: torch.Tensor,
    device: torch.device,
    jacobian_type: str,
    probes: int,
    seed: int,
    chunk_size: int = 8192,
    return_endpoint: bool = False,
    tolerance: Optional[float] = None,
):
    """Sensitivity of the teacher's endpoint theta_1 to its noise, J = d theta_1 / d noise, in one of three forms.

    Tangent equations are integrated jointly with the teacher ODE (same solver and tolerances as teacher
    sampling), costing Jacobian-vector products (JVPs) of the vector field per step. With return_endpoint the
    endpoints theta_1 are returned too, so trajectory generation gets theta and J from a single solve:
      "full":     J itself via the variational equation dJ/dt = grad_theta v J (d JVPs per step) -> (n, d, d).
      "diagonal": Hutchinson estimate of diag(J J^T) = per-dimension posterior spread s_i^2, from `probes`
                  forward tangents dT/dt = grad_theta v T with T(0) = V ~ N(0, I): mean_k (J v_k)^2 -> (n, d).
      "trace":    Hutchinson estimate of log|det J| = int tr(grad_theta v) dt with one Rademacher probe per
                  trajectory (1 JVP per step) -> (n,).
      "finite_difference_trace": the same estimate with the probe's directional derivative taken by a forward
                  difference, (v(theta + h eps) - v(theta)) / h, from one forward pass over a doubled batch instead
                  of a JVP (much cheaper on MPS for large networks); same probes as "trace" -> (n,).
      "vjp_sketch": `probes` adjoint probes mu = J^-T r, r ~ N(0, I), integrated forward with
                  d mu/dt = -(grad_theta v)^T mu (one batched vector-Jacobian product per step). Since
                  E[mu mu^T] = J^-T J^-1, (1/k) sum (mu_i . e)^2 is an unbiased estimate of the full pull-back
                  form e^T J^-T J^-1 e -> (n, d, k).
    `tolerance` overrides the teacher's atol/rtol (used for "vjp_sketch", whose estimate is noisy anyway).
    """
    from torchdiffeq import odeint

    flow_config = teacher_model.model_config
    dim = noise.shape[1]
    ode_options = {"dtype": torch.float32} if device.type == "mps" else {}
    generator = torch.Generator().manual_seed(seed)
    outputs, endpoints = [], []
    teacher_model.eval()
    pass_start = time.time()
    for start in range(0, len(noise), chunk_size):
        noise_chunk = noise[start:start + chunk_size].to(device)
        context_chunk = context[start:start + chunk_size].to(device)
        batch = len(noise_chunk)
        dtype = noise_chunk.dtype
        field_at = lambda time: (lambda theta: teacher_model.forward(time, theta, context_chunk))

        if jacobian_type == "vjp_sketch":
            sketch_probes = torch.randn(batch, dim, int(probes), generator=generator).to(device=device, dtype=dtype)

            def rhs(time, state):
                theta_t, adjoint = state
                with torch.enable_grad():
                    velocity, pullback = torch.func.vjp(field_at(time), theta_t)
                    transposed = torch.func.vmap(lambda column: pullback(column)[0], in_dims=2, out_dims=2)(adjoint)
                return velocity, -transposed

            initial = (noise_chunk, sketch_probes)
        elif jacobian_type in ("trace", "finite_difference_trace"):
            probe = (torch.randint(0, 2, (batch, dim), generator=generator) * 2 - 1).to(device=device, dtype=dtype)
            if jacobian_type == "trace":

                def rhs(time, state):
                    theta_t, _ = state
                    velocity, directional = torch.func.jvp(field_at(time), (theta_t,), (probe,))
                    return velocity, (probe * directional).sum(-1)

            else:
                doubled_context = torch.cat([context_chunk, context_chunk], dim=0)
                step = _FINITE_DIFFERENCE_STEP

                def rhs(time, state):
                    theta_t, _ = state
                    both = teacher_model.forward(time, torch.cat([theta_t, theta_t + step * probe]), doubled_context)
                    velocity, shifted = both[:batch], both[batch:]
                    return velocity, (probe * (shifted - velocity)).sum(-1) / step

            initial = (noise_chunk, torch.zeros(batch, device=device, dtype=dtype))
        else:
            columns = dim if jacobian_type == "full" else int(probes)
            if jacobian_type == "full":
                tangents = torch.eye(dim, device=device, dtype=dtype).expand(batch, dim, dim).contiguous()
            else:
                tangents = torch.randn(batch, dim, columns, generator=generator).to(device=device, dtype=dtype)

            def rhs(time, state):
                theta_t, tangent = state
                field = field_at(time)
                # The first JVP also returns the velocity, so the field is not evaluated separately.
                velocity, first = torch.func.jvp(field, (theta_t,), (tangent[:, :, 0],))
                pushed = [first] + [torch.func.jvp(field, (theta_t,), (tangent[:, :, k],))[1] for k in range(1, columns)]
                return velocity, torch.stack(pushed, dim=-1)

            initial = (noise_chunk, tangents)

        t_span = torch.tensor([0.0, 1.0 - flow_config.sigma_min], device=device, dtype=dtype)
        with torch.no_grad():
            atol = flow_config.atol if tolerance is None else tolerance
            rtol = flow_config.rtol if tolerance is None else tolerance
            theta_path, result = odeint(rhs, initial, t_span, atol=atol, rtol=rtol, method="dopri5", options=ode_options)
        result = result[-1]
        if jacobian_type == "diagonal":
            result = result.pow(2).mean(-1)
        outputs.append(result.cpu())
        endpoints.append(theta_path[-1])
        if not return_endpoint:
            _report_progress(f"{jacobian_type} Jacobian pass", start + batch, len(noise), pass_start)
    if return_endpoint:
        return torch.cat(endpoints), torch.cat(outputs)
    return torch.cat(outputs)


def _sensitivity_paths(trajectory_dir: Path, jacobian_type: str) -> Tuple[Path, Path]:
    return trajectory_dir / f"endpoint_jacobian_{jacobian_type}.npy", trajectory_dir / _JACOBIAN_METADATA_FILENAME


def _sensitivity_metadata(
    trajectory_metadata: Dict[str, Any],
    jacobian_type: str,
    probes: int,
    tolerance: Optional[float] = None,
) -> Dict[str, Any]:
    metadata = {
        "trajectories": json.loads(json.dumps(trajectory_metadata, sort_keys=True)),
        "jacobian_type": jacobian_type,
        "probes": None if jacobian_type == "full" else int(probes),
    }
    if jacobian_type == "vjp_sketch":
        metadata["tolerance"] = float(tolerance)
    return metadata


def _save_flow_sensitivity(
    trajectory_dir: Path,
    trajectory_metadata: Dict[str, Any],
    jacobian_type: str,
    probes: int,
    values: torch.Tensor,
    tolerance: Optional[float] = None,
) -> None:
    values_path, metadata_path = _sensitivity_paths(trajectory_dir, jacobian_type)
    np.save(values_path, values.numpy())
    existing = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.exists() else {}
    metadata = existing if isinstance(existing, dict) and "trajectories" not in existing else {}
    metadata[jacobian_type] = _sensitivity_metadata(trajectory_metadata, jacobian_type, probes, tolerance)
    metadata_path.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


def _cached_flow_sensitivity(
    teacher_bundle: "TeacherTrajectoryBundle",
    teacher_model,
    device: torch.device,
    jacobian_type: str,
    probes: int,
    tolerance: Optional[float] = None,
    chunk_size: int = 8192,
) -> torch.Tensor:
    """Flow sensitivity for every cached teacher pair (row order of the trajectory cache).

    Normally written by trajectory generation in the same solve as theta; computed here in a separate pass only
    when the trajectories are cached without it (e.g. after switching jacobian_type).
    """
    trajectory_dir = teacher_bundle.trajectory_dir
    values_path, metadata_path = _sensitivity_paths(trajectory_dir, jacobian_type)
    trajectory_metadata_path = trajectory_dir / _CACHE_METADATA_FILENAME
    trajectory_metadata = (
        json.loads(trajectory_metadata_path.read_text(encoding="utf-8")) if trajectory_metadata_path.exists() else None
    )
    if trajectory_metadata is not None and values_path.exists() and metadata_path.exists():
        cached = json.loads(metadata_path.read_text(encoding="utf-8"))
        if isinstance(cached, dict) and cached.get(jacobian_type) == _sensitivity_metadata(
            trajectory_metadata, jacobian_type, probes, tolerance
        ):
            return torch.from_numpy(np.load(values_path))
    train, val = teacher_bundle.train_dataset, teacher_bundle.val_dataset
    total = len(train) + len(val)
    noise = torch.empty(total, train.noise_state.shape[1], dtype=train.noise_state.dtype)
    context = torch.empty(total, train.context.shape[1], dtype=train.context.dtype)
    for split in (train, val):
        noise[split.source_indices] = split.noise_state
        context[split.source_indices] = split.context
    values = _teacher_flow_sensitivity(
        teacher_model, noise, context, device, jacobian_type, probes, seed=0, chunk_size=chunk_size,
        tolerance=tolerance if jacobian_type == "vjp_sketch" else None,
    )
    if trajectory_metadata is not None:
        _save_flow_sensitivity(trajectory_dir, trajectory_metadata, jacobian_type, probes, values, tolerance)
    return values


def attach_pullback_endpoint_metric(
    teacher_bundle: "TeacherTrajectoryBundle",
    teacher_model,
    device: torch.device,
    weight: float = 1.0,
    epsilon_fraction: float = 0.0,
    jacobian_type: str = "full",
    probes: int = 4,
    sketch_tolerance: float = 1e-3,
    chunk_size: int = 8192,
    sketch_batch_size: int = 512,
) -> float:
    """Give each teacher pair the endpoint metric I + weight * M, with M the (normalised) pull-back of the flow.

    The endpoint loss e^T (I + weight * M) e is the plain squared error plus `weight` times the error measured
    relative to the teacher's local posterior width (to first order, the error in the teacher's noise
    coordinates). The MSE term keeps every direction weighted at least 1. M by `jacobian_type`:
      "full":     (J J^T + eps I)^-1, a (d, d) matrix per pair (exact; O(d^2) memory, for low dimensions).
      "diagonal": diag(1 / (s_i^2 + eps)) with s_i^2 the Hutchinson estimate of diag(J J^T), stored as (d,).
      "trace":    1 / (sigma_g^2 + eps) with sigma_g = |det J|^(1/d) the geometric-mean local width, a scalar.
      "vjp_sketch": (1/k) sum mu_i mu_i^T with mu_i = J^-T r_i, an unbiased rank-k estimate of J^-T J^-1, stored
                  as low-rank factors (no inversion, so eps does not apply).
    eps = (epsilon_fraction * median reference width)^2 caps weights of very thin directions (0 = no cap); the
    reference width is sigma_max(J) (full), max_i s_i (diagonal) or sigma_g (trace). M is normalised to mean
    trace / d = 1 on the training split, so at weight 1 both terms contribute equally on average.
    Returns the time taken.
    """
    if jacobian_type not in JACOBIAN_TYPES:
        raise ValueError(f"tensorproduct_koopman.jacobian_type must be one of {JACOBIAN_TYPES}, got {jacobian_type!r}.")
    start_time = time.time()
    all_values = _cached_flow_sensitivity(
        teacher_bundle,
        teacher_model,
        device,
        jacobian_type,
        probes,
        tolerance=sketch_tolerance,
        # vjp_sketch's batched VJPs need far more memory per pair than a plain solve, so it has its own batch size.
        chunk_size=sketch_batch_size if jacobian_type == "vjp_sketch" else chunk_size,
    )
    train_values = all_values[teacher_bundle.train_dataset.source_indices]
    val_values = all_values[teacher_bundle.val_dataset.source_indices]
    dim = teacher_bundle.train_dataset.theta.shape[1]
    if jacobian_type == "vjp_sketch":
        train_factors, val_factors = train_values, val_values
        if not (torch.isfinite(train_factors).all() and torch.isfinite(val_factors).all()):
            raise FloatingPointError("vjp_sketch probes are not finite; lower tensorproduct_koopman.vjp_sketch_tolerance.")
        # mean trace / d of (1/k) mu mu^T is mean ||mu||^2 / (k d); scale factors so the metric term has mean 1.
        normaliser = train_factors.pow(2).sum(dim=(1, 2)).mean() / (train_factors.shape[2] * dim)
        scale = (float(weight) / (normaliser * train_factors.shape[2])).sqrt()
        train, val = teacher_bundle.train_dataset, teacher_bundle.val_dataset
        train.endpoint_metric = {"low_rank_factors": (scale * train_factors).to(train.theta.dtype)}
        val.endpoint_metric = {"low_rank_factors": (scale * val_factors).to(val.theta.dtype)}
        return time.time() - start_time
    if jacobian_type == "full":
        identity = torch.eye(dim, dtype=train_values.dtype)
        epsilon = (float(epsilon_fraction) * torch.linalg.svdvals(train_values)[:, 0].median()) ** 2
        build = lambda jacobian: torch.linalg.inv(jacobian @ jacobian.transpose(1, 2) + epsilon * identity)
        mean_trace = lambda metric: metric.diagonal(dim1=1, dim2=2).sum(-1).mean() / dim
        with_mse = lambda metric: identity + metric
    elif jacobian_type == "diagonal":
        epsilon = (float(epsilon_fraction) * train_values.clamp_min(0).sqrt().amax(-1).median()) ** 2
        build = lambda spread: 1.0 / (spread + epsilon)
        mean_trace = lambda metric: metric.mean()
        with_mse = lambda metric: 1.0 + metric
    else:  # "trace" and "finite_difference_trace" both estimate log|det J|
        geometric_width = lambda log_det: torch.exp(log_det / dim)
        epsilon = (float(epsilon_fraction) * geometric_width(train_values).median()) ** 2
        build = lambda log_det: 1.0 / (geometric_width(log_det) ** 2 + epsilon)
        mean_trace = lambda metric: metric.mean()
        with_mse = lambda metric: 1.0 + metric
    train_metric, val_metric = build(train_values), build(val_values)
    if not (torch.isfinite(train_metric).all() and torch.isfinite(val_metric).all()):
        raise FloatingPointError(
            "Pull-back metric is not finite (near-singular flow Jacobian); set "
            "tensorproduct_koopman.pullback_metric_epsilon > 0."
        )
    normaliser = mean_trace(train_metric)
    train, val = teacher_bundle.train_dataset, teacher_bundle.val_dataset
    train.endpoint_metric = with_mse(float(weight) * train_metric / normaliser).to(train.theta.dtype)
    val.endpoint_metric = with_mse(float(weight) * val_metric / normaliser).to(val.theta.dtype)
    return time.time() - start_time


@dataclass
class TeacherTrajectoryBundle:
    train_dataset: TeacherTrajectoryDataset
    val_dataset: TeacherTrajectoryDataset
    trajectory_dir: Path
    generation_time_seconds: float
    loaded_from_cache: bool
    num_samples: int
    num_context: int


def _load_trajectories(
    trajectory_dir: Path,
    load_full_trajectory: bool,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    noise_state = torch.tensor(np.load(trajectory_dir / "noise.npy"), dtype=torch.float32)
    theta = torch.tensor(np.load(trajectory_dir / "theta.npy"), dtype=torch.float32)
    context = torch.tensor(np.load(trajectory_dir / "context.npy"), dtype=torch.float32)
    time_grid = None
    path = None
    path_mask = None
    if load_full_trajectory:
        time_grid = torch.tensor(np.load(trajectory_dir / "time.npy"), dtype=torch.float32)
        path = torch.tensor(np.load(trajectory_dir / "path.npy"), dtype=torch.float32)
        path_mask = torch.tensor(np.load(trajectory_dir / "path_mask.npy"), dtype=torch.bool)
    return noise_state, theta, context, time_grid, path, path_mask


def _save_trajectories(
    trajectory_dir: Path,
    noise_state: torch.Tensor,
    theta: torch.Tensor,
    context: torch.Tensor,
    time_grid: Optional[torch.Tensor] = None,
    path: Optional[torch.Tensor] = None,
    path_mask: Optional[torch.Tensor] = None,
) -> None:
    trajectory_dir.mkdir(parents=True, exist_ok=True)
    np.save(trajectory_dir / "noise.npy", noise_state.numpy())
    np.save(trajectory_dir / "theta.npy", theta.numpy())
    np.save(trajectory_dir / "context.npy", context.numpy())
    if time_grid is not None and path is not None and path_mask is not None:
        np.save(trajectory_dir / "time.npy", time_grid.numpy())
        np.save(trajectory_dir / "path.npy", path.numpy())
        np.save(trajectory_dir / "path_mask.npy", path_mask.numpy())


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
    teacher_metadata: Dict[str, Any] = {
        "checkpoint_sha256": _teacher_model_digest(teacher_model, teacher_checkpoint_path),
        "num_samples": config.teacher.num_samples,
        "num_context": config.teacher.num_context,
        "batch_size": config.teacher.batch_size,
    }
    if config.teacher.store_full_trajectory:
        teacher_metadata.update(
            {
                "store_full_trajectory": True,
                "trajectory_steps": config.teacher.trajectory_steps,
            }
        )
    return {
        "version": _TRAJECTORY_CACHE_VERSION,
        "task": {
            "name": config.task.name,
            "seed": config.task.seed,
            "num_train_samples": config.task.num_train_samples,
            "train_fraction": config.task.train_fraction,
        },
        "teacher": teacher_metadata,
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


def _report_progress(label: str, done: int, total: int, start_time: float) -> None:
    """One line per batch: count done, elapsed minutes and a linear estimate of the time left."""
    elapsed = time.time() - start_time
    remaining = elapsed / done * (total - done) if done else float("nan")
    print(f"[teacher] {label}: {done}/{total} ({100 * done / total:.0f}%), "
          f"{elapsed / 60:.1f} min elapsed, ~{remaining / 60:.1f} min left", flush=True)


def _save_cache_metadata(trajectory_dir: Path, metadata: Dict[str, Any]) -> None:
    with open(trajectory_dir / _CACHE_METADATA_FILENAME, "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2, sort_keys=True)


def _split_teacher_data(
    noise_state: torch.Tensor,
    theta: torch.Tensor,
    context: torch.Tensor,
    time_grid: Optional[torch.Tensor],
    path: Optional[torch.Tensor],
    path_mask: Optional[torch.Tensor],
    include_full_trajectory: bool,
    train_fraction: float,
    seed: int,
) -> Tuple[TeacherTrajectoryDataset, TeacherTrajectoryDataset]:
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(len(theta), generator=generator)
    num_train = int(len(theta) * train_fraction)
    train_indices = permutation[:num_train]
    val_indices = permutation[num_train:]
    datasets = (
        TeacherTrajectoryDataset(
            noise_state[train_indices],
            theta[train_indices],
            context[train_indices],
            time_grid[train_indices] if time_grid is not None else None,
            path[train_indices] if path is not None else None,
            path_mask[train_indices] if path_mask is not None else None,
            include_full_trajectory,
        ),
        TeacherTrajectoryDataset(
            noise_state[val_indices],
            theta[val_indices],
            context[val_indices],
            time_grid[val_indices] if time_grid is not None else None,
            path[val_indices] if path is not None else None,
            path_mask[val_indices] if path_mask is not None else None,
            include_full_trajectory,
        ),
    )
    datasets[0].source_indices, datasets[1].source_indices = train_indices, val_indices
    return datasets


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
    include_full_trajectory: bool = False,
    sensitivity: Optional[Tuple[str, int]] = None,
) -> TeacherTrajectoryBundle:
    """Load cached teacher pairs or generate them.

    sensitivity=(jacobian_type, probes) also computes the flow sensitivity used by the pull-back endpoint
    metric in the same augmented solve that produces theta (no second integration), and caches it alongside.
    """
    trajectory_dir = resolve_teacher_dir(
        config.logging.output_root,
        config.task.name,
        config.teacher.trajectory_dir,
    )
    noise_path = trajectory_dir / "noise.npy"
    theta_path = trajectory_dir / "theta.npy"
    context_path = trajectory_dir / "context.npy"
    time_path = trajectory_dir / "time.npy"
    path_path = trajectory_dir / "path.npy"
    path_mask_path = trajectory_dir / "path_mask.npy"
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
        and (
            not config.teacher.store_full_trajectory
            or (time_path.exists() and path_path.exists() and path_mask_path.exists())
        )
        and _cache_matches(trajectory_dir, cache_metadata)
    ):
        noise_state, theta, context, time_grid, path, path_mask = _load_trajectories(
            trajectory_dir,
            load_full_trajectory=config.teacher.store_full_trajectory,
        )
        loaded_from_cache = True
    else:
        if not config.teacher.generate_trajectories:
            raise FileNotFoundError(
                "Teacher trajectories were not found in the cache, and "
                "teacher.generate_trajectories is false. Enable teacher.generate_trajectories "
                f"or provide a valid cached trajectory directory at {trajectory_dir}."
            )
        loaded_from_cache = False
        context_pool = _select_context_pool(dataset_bundle, config.teacher.num_context, config.task.seed)
        teacher_model.eval()
        noise_batches = []
        sensitivity_batches = []
        theta_batches = []
        context_batches = []
        path_batches = []
        time_batches = []
        path_mask_batches = []
        time_grid = None
        total_samples = config.teacher.num_samples
        batch_size = config.teacher.batch_size
        generation_start = time.time()
        with torch.no_grad():
            for start in range(0, total_samples, batch_size):
                current_batch = min(batch_size, total_samples - start)
                if total_samples <= len(context_pool):
                    # Enough distinct contexts: use each pool context (already a random subset) exactly once.
                    indices = torch.arange(start, start + current_batch)
                else:
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
                if config.teacher.store_full_trajectory:
                    time_batch, path_batch, path_mask_batch = teacher_model.sample_trajectory(
                        context_batch,
                        initial_noise=noise_batch,
                        max_trajectory_steps=config.teacher.trajectory_steps,
                    )
                    valid_counts = path_mask_batch.long().sum(dim=1).clamp_min(1)
                    theta_batch = path_batch[
                        torch.arange(current_batch, device=path_batch.device),
                        valid_counts - 1,
                    ]
                    time_batches.append(time_batch.detach().cpu())
                    path_batches.append(path_batch.detach().cpu())
                    path_mask_batches.append(path_mask_batch.detach().cpu())
                elif sensitivity is not None and sensitivity[0] in _SINGLE_SOLVE_TYPES:
                    theta_batch, sensitivity_batch = _teacher_flow_sensitivity(
                        teacher_model,
                        noise_batch,
                        context_batch,
                        device,
                        sensitivity[0],
                        sensitivity[1],
                        seed=config.task.seed + start,
                        chunk_size=current_batch,
                        return_endpoint=True,
                    )
                    sensitivity_batches.append(sensitivity_batch)
                else:
                    theta_batch = teacher_model.sample_batch(context_batch, initial_noise=noise_batch)
                noise_batches.append(noise_batch.detach().cpu())
                theta_batches.append(theta_batch.detach().cpu())
                context_batches.append(context_batch.detach().cpu())
                _report_progress("trajectories", start + current_batch, total_samples, generation_start)
        noise_state = torch.cat(noise_batches, dim=0)
        theta = torch.cat(theta_batches, dim=0)
        context = torch.cat(context_batches, dim=0)
        path = torch.cat(path_batches, dim=0) if path_batches else None
        time_grid = torch.cat(time_batches, dim=0) if time_batches else None
        path_mask = torch.cat(path_mask_batches, dim=0) if path_mask_batches else None
        if config.teacher.cache_trajectories:
            _save_trajectories(trajectory_dir, noise_state, theta, context, time_grid, path, path_mask)
            _save_cache_metadata(trajectory_dir, cache_metadata)
            if sensitivity_batches:
                _save_flow_sensitivity(
                    trajectory_dir, cache_metadata, sensitivity[0], sensitivity[1], torch.cat(sensitivity_batches)
                )
    generation_time_seconds = time.time() - start_time

    train_dataset, val_dataset = _split_teacher_data(
        noise_state=noise_state,
        theta=theta,
        context=context,
        time_grid=time_grid,
        path=path,
        path_mask=path_mask,
        include_full_trajectory=include_full_trajectory,
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
