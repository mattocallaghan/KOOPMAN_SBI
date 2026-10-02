"""Evaluation for image tasks without reference posterior samples (e.g. the camera model).

With no ground-truth posterior, a model is judged two ways per observation:
- against the teacher flow: C2ST between the model's samples and the teacher's (teacher samples are cached
  per teacher checkpoint); the flow evaluated against its own cached samples should score ~0.5;
- against the true image (point metrics): MSE / PSNR / SSIM of the posterior mean, the mean per-sample MSE,
  the mean pixelwise posterior std, and the fraction of pixels whose true value lies in the central 90%
  posterior interval (0.9 for a pixelwise-calibrated posterior).
Samples are clipped to the prior's pixel range instead of being rejected.
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from koopman_sbi.config import ExperimentConfig
from koopman_sbi.data import DatasetBundle
from koopman_sbi.logging_utils import ExperimentLogger
from koopman_sbi.tasks import get_task

IMAGE_METRICS = [
    "c2st_vs_teacher",
    "posterior_mean_mse",
    "posterior_mean_psnr",
    "posterior_mean_ssim",
    "sample_mse",
    "posterior_std",
    "coverage_90",
]


def _synchronize(device: torch.device) -> None:
    if device.type == "mps":
        torch.mps.synchronize()
    elif device.type == "cuda":
        torch.cuda.synchronize(device)


def c2st_torch(first: torch.Tensor, second: torch.Tensor, device: torch.device, folds: int = 5, seed: int = 0) -> float:
    """Classifier two-sample test: k-fold accuracy of an MLP separating the two sample sets (0.5 = indistinguishable).

    A torch MLP (d -> 256 -> 256 -> 1) on z-scored features; returns max(acc, 1 - acc).
    """
    generator = torch.Generator().manual_seed(seed)
    n = min(len(first), len(second))
    data = torch.cat([first[:n], second[:n]]).float()
    labels = torch.cat([torch.zeros(n), torch.ones(n)])
    data = (data - data.mean(0)) / data.std(0).clamp_min(1e-6)
    permutation = torch.randperm(2 * n, generator=generator)
    data, labels = data[permutation].to(device), labels[permutation].to(device)
    fold_size = (2 * n) // folds
    accuracies = []
    for fold in range(folds):
        test = torch.zeros(2 * n, dtype=torch.bool, device=device)
        test[fold * fold_size:(fold + 1) * fold_size] = True
        torch.manual_seed(seed + fold)
        classifier = nn.Sequential(
            nn.Linear(data.shape[1], 256), nn.SiLU(), nn.Linear(256, 256), nn.SiLU(), nn.Linear(256, 1)
        ).to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3, weight_decay=1e-4)
        train_x, train_y = data[~test], labels[~test]
        for _ in range(60):
            order = torch.randperm(len(train_x), device=device)
            for start in range(0, len(order), 256):
                idx = order[start:start + 256]
                loss = F.binary_cross_entropy_with_logits(classifier(train_x[idx]).squeeze(-1), train_y[idx])
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
        with torch.no_grad():
            predictions = (classifier(data[test]).squeeze(-1) > 0).float()
            accuracies.append((predictions == labels[test]).float().mean().item())
    accuracy = float(np.mean(accuracies))
    return max(accuracy, 1.0 - accuracy)


def _ssim(prediction: torch.Tensor, target: torch.Tensor, side: int, data_range: float = 1.0) -> float:
    """Mean SSIM (Gaussian window 11, sigma 1.5) between two flat images."""
    coords = torch.arange(11, dtype=torch.float64) - 5
    window_1d = torch.exp(-coords**2 / (2 * 1.5**2))
    window = (window_1d[:, None] * window_1d[None, :])
    window = (window / window.sum()).view(1, 1, 11, 11)
    x = prediction.double().view(1, 1, side, side)
    y = target.double().view(1, 1, side, side)
    blur = lambda image: F.conv2d(F.pad(image, (5, 5, 5, 5), mode="reflect"), window)
    mu_x, mu_y = blur(x), blur(y)
    var_x, var_y, cov = blur(x * x) - mu_x**2, blur(y * y) - mu_y**2, blur(x * y) - mu_x * mu_y
    c1, c2 = (0.01 * data_range) ** 2, (0.03 * data_range) ** 2
    ssim_map = ((2 * mu_x * mu_y + c1) * (2 * cov + c2)) / ((mu_x**2 + mu_y**2 + c1) * (var_x + var_y + c2))
    return float(ssim_map.mean())


def _teacher_checkpoint(config: ExperimentConfig) -> Path:
    if config.teacher.checkpoint_path and Path(config.teacher.checkpoint_path).exists():
        return Path(config.teacher.checkpoint_path)
    return Path(config.logging.output_root) / config.task.name / "last_model" / "train_flow" / "best_model.pt"


def teacher_posterior_samples(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    observation: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    """Teacher-flow posterior samples (raw theta, clipped) for one observation, cached per teacher checkpoint."""
    from koopman_sbi.models.flow_matching import ConditionalFlowMatching

    task = get_task(config.task.name)
    checkpoint = _teacher_checkpoint(config)
    if not checkpoint.exists():
        raise FileNotFoundError(f"C2ST against the teacher needs a trained teacher; none found at {checkpoint}.")
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()[:16]
    cache_dir = Path(config.logging.output_root) / config.task.name / "shared" / "teacher_posterior_samples" / digest
    cache_path = cache_dir / f"observation_{observation:02d}_{num_samples}.npy"
    if cache_path.exists():
        return torch.from_numpy(np.load(cache_path))
    teacher = ConditionalFlowMatching.load(str(checkpoint), device=device)
    context = dataset_bundle.standardizer.standardize_x(task.get_observation(observation).float())
    generator = torch.Generator().manual_seed(10_000 + observation)
    noise = torch.randn(num_samples, dataset_bundle.dim_theta, generator=generator)
    samples = []
    with torch.no_grad():
        for start in range(0, num_samples, 500):
            chunk = noise[start:start + 500].to(device)
            samples.append(teacher.sample_batch(context.repeat(len(chunk), 1).to(device), initial_noise=chunk).cpu())
    samples = _to_prior_range(task, dataset_bundle.standardizer.inverse_theta(torch.cat(samples)))
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, samples.numpy())
    return samples


def _to_prior_range(task: Any, samples: torch.Tensor) -> torch.Tensor:
    low, high = getattr(task, "parameter_range", (None, None))
    return samples.clamp(low, high) if low is not None else samples


def _plot_image_posteriors(
    rows: List[Dict[str, torch.Tensor]],
    side: int,
    output_path: Path,
    title: str,
) -> None:
    columns = ["truth", "observation", "posterior mean", "posterior std", "sample 1", "sample 2", "sample 3"]
    fig, axes = plt.subplots(len(rows), len(columns), figsize=(1.6 * len(columns), 1.6 * len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        images = [row["truth"], row["observation"], row["mean"], row["std"], *row["samples"][:3]]
        for c, image in enumerate(images):
            ax = axes[r][c]
            vmax = None if columns[c] == "posterior std" else 1.0
            ax.imshow(image.reshape(side, side).numpy(), cmap="Greys", vmin=0.0, vmax=vmax)
            ax.set_xticks([])
            ax.set_yticks([])
            if r == 0:
                ax.set_title(columns[c], fontsize=8)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(output_path, dpi=120)
    plt.close(fig)


def evaluate_image_model(
    model,
    model_name: str,
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    output_dir: Path,
    sample_kwargs: Dict[str, Any] | None = None,
    logger: ExperimentLogger | None = None,
) -> Dict[str, object]:
    """Image-task counterpart of evaluation.evaluate_model; returns the same result structure."""
    task = get_task(config.task.name)
    sample_kwargs = sample_kwargs or {}
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    side = int(round(dataset_bundle.dim_theta ** 0.5))
    num_samples = int(config.evaluation.num_posterior_samples)
    standardizer = dataset_bundle.standardizer

    warmup = standardizer.standardize_x(task.get_observation(config.evaluation.observations[0]).float())
    with torch.no_grad():
        model.sample_batch(warmup.repeat(2, 1), **sample_kwargs)
    _synchronize(model.device)

    per_observation, plot_rows, speed_values = [], [], []
    posterior_cache: Dict[int, torch.Tensor] = {}
    reference_cache: Dict[int, torch.Tensor] = {}
    for obs in config.evaluation.observations:
        observation = task.get_observation(obs).float()
        truth = task.get_true_parameters(obs).float().reshape(-1)
        context = standardizer.standardize_x(observation).repeat(num_samples, 1)
        _synchronize(model.device)
        start_time = time.perf_counter()
        with torch.no_grad():
            standardized = model.sample_batch(context, **sample_kwargs)
        _synchronize(model.device)
        sampling_time_ms = (time.perf_counter() - start_time) * 1000.0
        samples = _to_prior_range(task, standardizer.inverse_theta(standardized.detach().cpu()))
        teacher_samples = teacher_posterior_samples(config, dataset_bundle, obs, num_samples, model.device)

        mean, std = samples.mean(0), samples.std(0)
        low, high = samples.quantile(0.05, dim=0), samples.quantile(0.95, dim=0)
        mean_mse = float((mean - truth).pow(2).mean())
        row = {
            "observation": obs,
            "sampling_time_ms": sampling_time_ms,
            "acceptance_rate": 1.0,
            "num_generated_samples": num_samples,
            "num_retained_samples": num_samples,
            "num_reference_samples": len(teacher_samples),
            "num_model_samples": num_samples,
            "sampling_time_per_generated_sample_ms": sampling_time_ms / num_samples,
            "sampling_time_per_retained_sample_ms": sampling_time_ms / num_samples,
            "c2st_vs_teacher": c2st_torch(samples, teacher_samples, model.device),
            "posterior_mean_mse": mean_mse,
            "posterior_mean_psnr": float(10.0 * np.log10(1.0 / max(mean_mse, 1e-12))),
            "posterior_mean_ssim": _ssim(mean, truth, side),
            "sample_mse": float((samples - truth).pow(2).mean()),
            "posterior_std": float(std.mean()),
            "coverage_90": float(((truth >= low) & (truth <= high)).float().mean()),
        }
        per_observation.append(row)
        speed_values.append(sampling_time_ms)
        posterior_cache[obs] = samples
        reference_cache[obs] = teacher_samples
        plot_rows.append({"truth": truth, "observation": observation.reshape(-1), "mean": mean, "std": std,
                          "samples": list(samples[:3])})

    if config.evaluation.save_observation_plots:
        plot_path = plot_dir / f"{model_name}_posteriors.png"
        _plot_image_posteriors(plot_rows, side, plot_path, f"{model_name}: {config.task.name}")
        if logger is not None:
            logger.log_image(f"{model_name}_posteriors", plot_path)

    summary = {
        "model_name": model_name,
        "mean_sampling_time_ms": float(np.mean(speed_values)),
        "std_sampling_time_ms": float(np.std(speed_values)),
        "mean_acceptance_rate": 1.0,
        "mean_num_generated_samples": float(num_samples),
        "mean_num_retained_samples": float(num_samples),
        "mean_sampling_time_per_generated_sample_ms": float(np.mean(speed_values)) / num_samples,
        "mean_sampling_time_per_retained_sample_ms": float(np.mean(speed_values)) / num_samples,
    }
    for metric_name in IMAGE_METRICS:
        summary[f"mean_{metric_name}"] = float(np.mean([row[metric_name] for row in per_observation]))

    from koopman_sbi.evaluation import _write_metrics

    _write_metrics(output_dir / f"{model_name}_per_observation.csv", per_observation)
    with open(output_dir / f"{model_name}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "summary": summary,
        "per_observation": per_observation,
        "posterior_cache": posterior_cache,
        "reference_cache": reference_cache,
    }
