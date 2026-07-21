from __future__ import annotations

import csv
from dataclasses import dataclass, field
import json
import math
import time
from pathlib import Path
from typing import Any, Dict, List, Mapping, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sbibm
import torch
from sbibm.metrics import c2st, ksd, median_distance, mmd, posterior_mean_error, posterior_variance_ratio

from koopman_sbi.config import ExperimentConfig
from koopman_sbi.data import DatasetBundle
from koopman_sbi.logging_utils import ExperimentLogger


def _c2st_canonicalized(posterior: torch.Tensor, reference: torch.Tensor) -> float:
    raw_score = float(c2st(posterior, reference).item())
    return max(raw_score, 1.0 - raw_score)


SAMPLE_METRICS = {
    "c2st": lambda task, obs, posterior, reference: _c2st_canonicalized(posterior, reference),
    "mmd": lambda task, obs, posterior, reference: mmd(posterior, reference).item(),
    "posterior_mean_error": lambda task, obs, posterior, reference: posterior_mean_error(posterior, reference).item(),
    "posterior_variance_ratio": lambda task, obs, posterior, reference: posterior_variance_ratio(posterior, reference).item(),
    "median_distance": lambda task, obs, posterior, reference: median_distance(posterior, reference).item(),
    "ksd": lambda task, obs, posterior, reference: ksd(task, obs, posterior),
}


@dataclass
class BenchmarkModelSpec:
    label: str
    model: Any
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)
    timing_metadata: Dict[str, float | int | bool | str] = field(default_factory=dict)


def _filter_to_prior_support(task, samples: torch.Tensor, device: torch.device) -> Tuple[torch.Tensor, float]:
    if device.type == "mps":
        log_prob = task.prior_dist.log_prob(samples.cpu())
        mask = torch.isfinite(log_prob)
        mask_device = mask.to(device)
    else:
        mask_device = torch.isfinite(task.prior_dist.log_prob(samples))
    filtered_samples = samples[mask_device]
    acceptance_rate = float(mask_device.float().mean().item())
    return filtered_samples, acceptance_rate


def _plot_posterior_scatter(
    reference_samples: torch.Tensor,
    posterior_samples: torch.Tensor,
    output_path: Path,
    title: str,
) -> None:
    plt.figure(figsize=(8, 8))
    reference_np = reference_samples.detach().cpu().numpy()
    posterior_np = posterior_samples.detach().cpu().numpy()
    if reference_np.shape[1] == 1:
        plt.hist(reference_np[:, 0], bins=40, alpha=0.4, label="reference")
        plt.hist(posterior_np[:, 0], bins=40, alpha=0.4, label="model")
    else:
        plt.scatter(reference_np[:, 0], reference_np[:, 1], s=1.0, alpha=0.25, label="reference")
        plt.scatter(posterior_np[:, 0], posterior_np[:, 1], s=1.0, alpha=0.25, label="model")
    plt.legend()
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def evaluate_model(
    model,
    model_name: str,
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    output_dir: Path,
    sample_kwargs: Dict[str, Any] | None = None,
    logger: ExperimentLogger | None = None,
) -> Dict[str, object]:
    task = sbibm.get_task(config.task.name)
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = output_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    per_observation: List[Dict[str, float]] = []
    posterior_cache: Dict[int, torch.Tensor] = {}
    reference_cache: Dict[int, torch.Tensor] = {}
    speed_values: List[float] = []
    sample_kwargs = sample_kwargs or {}

    for obs in config.evaluation.observations:
        reference_samples = task.get_reference_posterior_samples(num_observation=obs)
        observation = task.get_observation(num_observation=obs).float()
        standardized_observation = dataset_bundle.standardizer.standardize_x(observation)
        context = standardized_observation.repeat((config.evaluation.num_posterior_samples * 2, 1))
        num_generated_samples = int(len(context))

        start_time = time.time()
        posterior_standardized = model.sample_batch(context, **sample_kwargs)
        sampling_time_ms = (time.time() - start_time) * 1000.0
        posterior_samples = dataset_bundle.standardizer.inverse_theta(posterior_standardized)
        posterior_samples, acceptance_rate = _filter_to_prior_support(task, posterior_samples, model.device)
        sample_count = min(len(reference_samples), len(posterior_samples))
        posterior_samples = posterior_samples[:sample_count]
        reference_samples = reference_samples[:sample_count]
        reference_samples = reference_samples.to(posterior_samples.device)
        posterior_samples_for_metrics = posterior_samples.detach().cpu()
        reference_samples_for_metrics = reference_samples.detach().cpu()

        metrics_row = {
            "observation": obs,
            "sampling_time_ms": sampling_time_ms,
            "acceptance_rate": acceptance_rate,
            "num_generated_samples": num_generated_samples,
            "num_retained_samples": sample_count,
            "num_reference_samples": len(reference_samples),
            "num_model_samples": len(posterior_samples),
            "sampling_time_per_generated_sample_ms": sampling_time_ms / max(num_generated_samples, 1),
            "sampling_time_per_retained_sample_ms": sampling_time_ms / max(sample_count, 1),
        }
        for metric_name in config.evaluation.metrics:
            if metric_name not in SAMPLE_METRICS:
                continue
            metrics_row[metric_name] = float(
                SAMPLE_METRICS[metric_name](
                    task,
                    obs,
                    posterior_samples_for_metrics,
                    reference_samples_for_metrics,
                )
            )
        per_observation.append(metrics_row)
        posterior_cache[obs] = posterior_samples.detach().cpu()
        reference_cache[obs] = reference_samples_for_metrics
        speed_values.append(sampling_time_ms)

        if config.evaluation.save_observation_plots:
            plot_path = plot_dir / f"{model_name}_observation_{obs:02d}.png"
            _plot_posterior_scatter(
                reference_samples,
                posterior_samples,
                plot_path,
                title=f"{model_name} observation {obs}",
            )
            if logger is not None:
                logger.log_image(f"{model_name}_observation_{obs:02d}", plot_path)

    summary = {
        "model_name": model_name,
        "mean_sampling_time_ms": float(np.mean(speed_values)),
        "std_sampling_time_ms": float(np.std(speed_values)),
        "mean_acceptance_rate": float(np.mean([row["acceptance_rate"] for row in per_observation])),
        "mean_num_generated_samples": float(np.mean([row["num_generated_samples"] for row in per_observation])),
        "mean_num_retained_samples": float(np.mean([row["num_retained_samples"] for row in per_observation])),
        "mean_sampling_time_per_generated_sample_ms": float(
            np.mean([row["sampling_time_per_generated_sample_ms"] for row in per_observation])
        ),
        "mean_sampling_time_per_retained_sample_ms": float(
            np.mean([row["sampling_time_per_retained_sample_ms"] for row in per_observation])
        ),
    }
    for metric_name in config.evaluation.metrics:
        metric_values = [row.get(metric_name) for row in per_observation if metric_name in row]
        if metric_values:
            summary[f"mean_{metric_name}"] = float(np.mean(metric_values))

    _write_metrics(output_dir / f"{model_name}_per_observation.csv", per_observation)
    with open(output_dir / f"{model_name}_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return {
        "summary": summary,
        "per_observation": per_observation,
        "posterior_cache": posterior_cache,
        "reference_cache": reference_cache,
    }


def benchmark_models(
    models: Dict[str, object] | List[BenchmarkModelSpec],
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    output_dir: Path,
    timing_metadata: Dict[str, Dict[str, float | int | bool]] | None = None,
    logger: ExperimentLogger | None = None,
) -> Dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    benchmark_specs = _normalize_benchmark_specs(models, timing_metadata)
    model_results = {}
    for spec in benchmark_specs:
        model_results[spec.label] = evaluate_model(
            model=spec.model,
            model_name=spec.label,
            config=config,
            dataset_bundle=dataset_bundle,
            output_dir=output_dir / spec.label,
            sample_kwargs=spec.sample_kwargs,
            logger=logger,
        )

    comparison_rows = []
    for observation in config.evaluation.observations:
        row = {"observation": observation}
        for model_name, result in model_results.items():
            observation_row = next(item for item in result["per_observation"] if item["observation"] == observation)
            for key, value in observation_row.items():
                if key == "observation":
                    continue
                row[f"{model_name}_{key}"] = value
        comparison_rows.append(row)

    _write_metrics(output_dir / "comparison_summary.csv", comparison_rows)
    with open(output_dir / "comparison_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {name: result["summary"] for name, result in model_results.items()},
            handle,
            indent=2,
        )
    _plot_spider_comparison(model_results, config.evaluation.metrics, output_dir / "benchmark_spider.png")
    _plot_side_by_side_posteriors(model_results, config.evaluation.observations, output_dir)
    if benchmark_specs:
        _write_benchmark_manifest(output_dir / "benchmark_manifest.json", benchmark_specs)
    if timing_metadata is not None:
        _write_worth_it_analysis(
            model_results=model_results,
            timing_metadata={spec.label: spec.timing_metadata for spec in benchmark_specs},
            teacher_num_samples=config.teacher.num_samples,
            output_dir=output_dir,
        )
    if logger is not None:
        logger.log_image("benchmark_spider", output_dir / "benchmark_spider.png")
        if timing_metadata is not None:
            logger.log_image("worth_it_curve", output_dir / "worth_it_curve.png")
            logger.log_image("worth_it_inference_curve", output_dir / "worth_it_inference_curve.png")
    return model_results


def _normalize_benchmark_specs(
    models: Dict[str, object] | List[BenchmarkModelSpec],
    timing_metadata: Dict[str, Dict[str, float | int | bool]] | None,
) -> List[BenchmarkModelSpec]:
    if isinstance(models, list):
        return models
    specs: List[BenchmarkModelSpec] = []
    for label, model in models.items():
        specs.append(
            BenchmarkModelSpec(
                label=label,
                model=model,
                timing_metadata=(timing_metadata or {}).get(label, {}),
            )
        )
    return specs


def _write_benchmark_manifest(path: Path, benchmark_specs: List[BenchmarkModelSpec]) -> None:
    manifest = [
        {
            "label": spec.label,
            "sample_kwargs": spec.sample_kwargs,
            "timing_metadata": spec.timing_metadata,
        }
        for spec in benchmark_specs
    ]
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)


def _write_metrics(path: Path, rows: List[Dict[str, float]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_spider_comparison(model_results: Dict[str, object], metric_names: List[str], output_path: Path) -> None:
    plotted_metrics = ["sampling_time_ms", *metric_names]
    metric_labels = {
        "sampling_time_ms": "sampling time",
        "posterior_mean_error": "mean error",
        "posterior_variance_ratio": "variance ratio",
    }
    raw_values: Dict[str, Dict[str, float]] = {}
    for metric_name in plotted_metrics:
        summary_key = "mean_sampling_time_ms" if metric_name == "sampling_time_ms" else f"mean_{metric_name}"
        values = {
            model_name: float(result["summary"][summary_key])
            for model_name, result in model_results.items()
            if summary_key in result["summary"]
        }
        if len(values) >= 2:
            raw_values[metric_name] = values
    if not raw_values:
        return

    transformed_values: Dict[str, Dict[str, float]] = {}
    axis_ranges: Dict[str, tuple[float, float]] = {}
    for metric_name, values in raw_values.items():
        transformed_values[metric_name] = {}
        if metric_name == "c2st":
            axis_ranges[metric_name] = (0.5, 1.0)
            for model_name, raw_value in values.items():
                transformed_values[metric_name][model_name] = float(np.clip(raw_value, 0.5, 1.0))
        elif metric_name == "posterior_mean_error":
            axis_ranges[metric_name] = (0.0, 1.0)
            for model_name, raw_value in values.items():
                transformed_values[metric_name][model_name] = float(np.clip(abs(raw_value), 0.0, 1.0))
        elif metric_name == "posterior_variance_ratio":
            axis_ranges[metric_name] = (0.0, 1.0)
            for model_name, raw_value in values.items():
                safe_value = max(raw_value, 1e-12)
                transformed_values[metric_name][model_name] = float(np.clip(abs(np.log(safe_value)), 0.0, 1.0))
        elif metric_name == "mmd":
            axis_ranges[metric_name] = (0.0, 0.1)
            for model_name, raw_value in values.items():
                transformed_values[metric_name][model_name] = float(np.clip(raw_value, 0.0, 0.1))
        elif metric_name == "sampling_time_ms":
            axis_ranges[metric_name] = (0.0, 3.0)
            for model_name, raw_value in values.items():
                transformed_values[metric_name][model_name] = float(np.clip(np.log10(max(raw_value, 1e-12)), 0.0, 3.0))
        else:
            metric_min = min(values.values())
            metric_max = max(values.values())
            axis_ranges[metric_name] = (metric_min, metric_max if metric_max > metric_min else metric_min + 1.0)
            for model_name, raw_value in values.items():
                transformed_values[metric_name][model_name] = float(raw_value)

    normalized_values: Dict[str, Dict[str, float]] = {}
    for metric_name, values in transformed_values.items():
        metric_min, metric_max = axis_ranges[metric_name]
        metric_range = metric_max - metric_min
        normalized_values[metric_name] = {}
        for model_name, raw_value in values.items():
            if metric_range <= 1e-12:
                score = 0.5
            else:
                score = (raw_value - metric_min) / metric_range
            normalized_values[metric_name][model_name] = float(score)

    model_names = list(model_results.keys())
    metric_order = [metric_name for metric_name in plotted_metrics if metric_name in normalized_values]
    num_metrics = len(metric_order)
    angles = np.linspace(0, 2 * math.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"polar": True})
    colors = {"flow_matching": "tab:blue", "koopman": "tab:orange", "npe": "tab:green"}
    linestyles = {"flow_matching": "-", "koopman": "--", "npe": "-."}
    for model_index, model_name in enumerate(model_names):
        if any(model_name not in normalized_values[metric_name] for metric_name in metric_order):
            continue
        scores = [normalized_values[metric_name][model_name] for metric_name in metric_order]
        scores += scores[:1]
        color = colors.get(model_name, plt.cm.tab10.colors[model_index % len(plt.cm.tab10.colors)])
        ax.plot(
            angles,
            scores,
            linewidth=2.5,
            linestyle=linestyles.get(model_name, "-"),
            marker="o",
            markersize=4,
            label=model_name,
            color=color,
            zorder=3 + model_index,
        )
        ax.fill(angles, scores, alpha=0.08, color=color, zorder=1)

    axis_labels = []
    for metric_name in metric_order:
        label = metric_labels.get(metric_name, metric_name.replace("_", " "))
        lower, upper = axis_ranges[metric_name]
        formatted_range = f"{lower:.3g}–{upper:.3g}"
        axis_labels.append(f"{label}\n({formatted_range})")
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(axis_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0.25", "0.50", "0.75", "1.00"])
    ax.set_title("Benchmark comparison (normalized; inner is better)", pad=24)
    ax.legend(loc="upper right", bbox_to_anchor=(1.25, 1.15))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_side_by_side_posteriors(model_results: Dict[str, object], observations: List[int], output_dir: Path) -> None:
    flow_label = _find_reference_label(model_results, ["flow_matching", "fmnpe_dopri5", "fmnpe_rk4_10"])
    koopman_label = _find_reference_label(model_results, ["koopman"])
    if flow_label is None or koopman_label is None:
        return
    for observation in observations:
        flow_samples = model_results[flow_label]["posterior_cache"][observation]
        koopman_samples = model_results[koopman_label]["posterior_cache"][observation]
        reference_samples = model_results[flow_label]["reference_cache"][observation]
        plt.figure(figsize=(14, 4))
        for index, (title, samples) in enumerate(
            [
                ("reference", reference_samples),
                (flow_label, flow_samples),
                (koopman_label, koopman_samples),
            ],
            start=1,
        ):
            plt.subplot(1, 3, index)
            sample_np = samples.numpy()
            if sample_np.shape[1] == 1:
                plt.hist(sample_np[:, 0], bins=40, alpha=0.5)
            else:
                plt.scatter(sample_np[:, 0], sample_np[:, 1], s=1.0, alpha=0.25)
            plt.title(title)
        plt.tight_layout()
        plt.savefig(output_dir / f"comparison_observation_{observation:02d}.png", dpi=150)
        plt.close()


def _find_reference_label(model_results: Mapping[str, object], candidates: List[str]) -> str | None:
    for candidate in candidates:
        if candidate in model_results:
            return candidate
    return None


def _write_worth_it_analysis(
    model_results: Dict[str, object],
    timing_metadata: Dict[str, Dict[str, float | int | bool]],
    teacher_num_samples: int,
    output_dir: Path,
) -> None:
    flow_label = _find_reference_label(model_results, ["flow_matching", "fmnpe_dopri5", "fmnpe_rk4_10"])
    koopman_label = _find_reference_label(model_results, ["koopman"])
    npe_label = _find_reference_label(model_results, ["npe"])
    if flow_label is None or koopman_label is None:
        return

    flow_meta = timing_metadata.get(flow_label, {})
    koopman_meta = timing_metadata.get(koopman_label, {})
    npe_meta = timing_metadata.get(npe_label, {}) if npe_label is not None else {}

    flow_training_time_seconds = float(flow_meta.get("training_time_seconds", 0.0))
    koopman_teacher_data_time_seconds = float(koopman_meta.get("teacher_data_time_seconds", 0.0))
    koopman_training_time_seconds = float(koopman_meta.get("training_time_seconds", 0.0))
    npe_training_time_seconds = float(npe_meta.get("training_time_seconds", 0.0))

    flow_sampling_time_per_observation_ms = float(model_results[flow_label]["summary"]["mean_sampling_time_ms"])
    koopman_sampling_time_per_observation_ms = float(model_results[koopman_label]["summary"]["mean_sampling_time_ms"])
    flow_parallel_generation_batch_size = int(model_results[flow_label]["summary"]["mean_num_generated_samples"])
    koopman_parallel_generation_batch_size = int(model_results[koopman_label]["summary"]["mean_num_generated_samples"])
    npe_sampling_time_per_observation_ms = (
        float(model_results[npe_label]["summary"]["mean_sampling_time_ms"]) if npe_label is not None else None
    )
    npe_parallel_generation_batch_size = (
        int(model_results[npe_label]["summary"]["mean_num_generated_samples"]) if npe_label is not None else None
    )

    max_num_observations = max(
        10**6,
        teacher_num_samples // max(flow_parallel_generation_batch_size, 1),
    )
    observation_counts = np.unique(
        np.round(np.geomspace(1, max_num_observations, num=100)).astype(int)
    )

    rows: List[Dict[str, float | int]] = []
    inference_only_rows: List[Dict[str, float | int]] = []
    koopman_crossover_count = None
    npe_crossover_count = None
    koopman_inference_only_crossover_count = None
    npe_inference_only_crossover_count = None
    for num_observations in observation_counts:
        flow_total_seconds = flow_training_time_seconds + (
            flow_sampling_time_per_observation_ms * num_observations / 1000.0
        )
        koopman_total_seconds = (
            flow_training_time_seconds
            + koopman_teacher_data_time_seconds
            + koopman_training_time_seconds
            + (koopman_sampling_time_per_observation_ms * num_observations / 1000.0)
        )
        row = {
            "num_observations": int(num_observations),
            "flow_total_time_seconds": float(flow_total_seconds),
            "koopman_total_time_seconds": float(koopman_total_seconds),
        }
        if koopman_crossover_count is None and koopman_total_seconds <= flow_total_seconds:
            koopman_crossover_count = int(num_observations)
        if npe_sampling_time_per_observation_ms is not None:
            npe_total_seconds = npe_training_time_seconds + (
                npe_sampling_time_per_observation_ms * num_observations / 1000.0
            )
            row["npe_total_time_seconds"] = float(npe_total_seconds)
            if npe_crossover_count is None and npe_total_seconds <= flow_total_seconds:
                npe_crossover_count = int(num_observations)
        rows.append(row)

        flow_inference_only_seconds = flow_sampling_time_per_observation_ms * num_observations / 1000.0
        koopman_inference_only_seconds = (
            koopman_teacher_data_time_seconds
            + (koopman_sampling_time_per_observation_ms * num_observations / 1000.0)
        )
        inference_only_row = {
            "num_observations": int(num_observations),
            "flow_inference_only_time_seconds": float(flow_inference_only_seconds),
            "koopman_inference_only_time_seconds": float(koopman_inference_only_seconds),
        }
        if (
            koopman_inference_only_crossover_count is None
            and koopman_inference_only_seconds <= flow_inference_only_seconds
        ):
            koopman_inference_only_crossover_count = int(num_observations)
        if npe_sampling_time_per_observation_ms is not None:
            npe_inference_only_seconds = npe_sampling_time_per_observation_ms * num_observations / 1000.0
            inference_only_row["npe_inference_only_time_seconds"] = float(npe_inference_only_seconds)
            if (
                npe_inference_only_crossover_count is None
                and npe_inference_only_seconds <= flow_inference_only_seconds
            ):
                npe_inference_only_crossover_count = int(num_observations)
        inference_only_rows.append(inference_only_row)

    worth_it_summary = {
        "teacher_num_samples": int(teacher_num_samples),
        "flow_training_time_seconds": flow_training_time_seconds,
        "koopman_teacher_data_time_seconds": koopman_teacher_data_time_seconds,
        "koopman_training_time_seconds": koopman_training_time_seconds,
        "npe_training_time_seconds": npe_training_time_seconds,
        "flow_mean_sampling_time_per_observation_ms": flow_sampling_time_per_observation_ms,
        "koopman_mean_sampling_time_per_observation_ms": koopman_sampling_time_per_observation_ms,
        "npe_mean_sampling_time_per_observation_ms": npe_sampling_time_per_observation_ms,
        "flow_parallel_generation_batch_size": flow_parallel_generation_batch_size,
        "koopman_parallel_generation_batch_size": koopman_parallel_generation_batch_size,
        "npe_parallel_generation_batch_size": npe_parallel_generation_batch_size,
        "koopman_intersection_num_observations": koopman_crossover_count,
        "npe_intersection_num_observations": npe_crossover_count,
        "koopman_inference_only_intersection_num_observations": koopman_inference_only_crossover_count,
        "npe_inference_only_intersection_num_observations": npe_inference_only_crossover_count,
    }
    _write_metrics(output_dir / "worth_it_curve.csv", rows)
    _write_metrics(output_dir / "worth_it_inference_curve.csv", inference_only_rows)
    with open(output_dir / "worth_it_summary.json", "w", encoding="utf-8") as handle:
        json.dump(worth_it_summary, handle, indent=2)
    _plot_worth_it_curve(rows, worth_it_summary, output_dir / "worth_it_curve.png")
    _plot_worth_it_inference_curve(
        inference_only_rows,
        worth_it_summary,
        output_dir / "worth_it_inference_curve.png",
    )


def _plot_worth_it_curve(
    rows: List[Dict[str, float | int]],
    worth_it_summary: Dict[str, float | int | None],
    output_path: Path,
) -> None:
    num_observations = [row["num_observations"] for row in rows]
    series = {
        "flow matching": [row["flow_total_time_seconds"] for row in rows],
        "koopman": [row["koopman_total_time_seconds"] for row in rows],
    }
    if "npe_total_time_seconds" in rows[0]:
        series["npe"] = [row["npe_total_time_seconds"] for row in rows]

    plt.figure(figsize=(8, 5))
    colors = {"flow matching": "tab:blue", "koopman": "tab:orange", "npe": "tab:green"}
    for label, values in series.items():
        plt.plot(num_observations, values, label=label, linewidth=2.0, color=colors.get(label))
    for model_key, row_key, training_key, slope_key, intersection_key, color in [
        (
            "koopman",
            "koopman_total_time_seconds",
            None,
            "koopman_mean_sampling_time_per_observation_ms",
            "koopman_intersection_num_observations",
            "tab:orange",
        ),
        (
            "npe",
            "npe_total_time_seconds",
            "npe_training_time_seconds",
            "npe_mean_sampling_time_per_observation_ms",
            "npe_intersection_num_observations",
            "tab:green",
        ),
    ]:
        if model_key == "npe" and "npe" not in series:
            continue
        crossover_count = worth_it_summary.get(intersection_key)
        if crossover_count is None:
            continue
        crossover_x = float(crossover_count)
        crossover_y = None
        for row in rows:
            if int(row["num_observations"]) == int(crossover_count):
                crossover_y = float(row[row_key])
                break
        if crossover_y is None:
            model_start = (
                float(worth_it_summary["flow_training_time_seconds"])
                + float(worth_it_summary["koopman_teacher_data_time_seconds"])
                + float(worth_it_summary["koopman_training_time_seconds"])
                if model_key == "koopman"
                else float(worth_it_summary[training_key])
            )
            model_slope = float(worth_it_summary[slope_key]) / 1000.0
            crossover_y = model_start + model_slope * crossover_x
        plt.scatter([crossover_x], [crossover_y], color=color, s=30, zorder=5)
        plt.annotate(
            f"{model_key}: {int(crossover_count)} obs",
            xy=(crossover_x, crossover_y),
            xytext=(8, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=color,
        )
    plt.xlabel("Number of independent observations")
    plt.ylabel("Total time (seconds, log scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Amortized total runtime comparison")
    flow_batch_size = worth_it_summary.get("flow_parallel_generation_batch_size")
    koopman_batch_size = worth_it_summary.get("koopman_parallel_generation_batch_size")
    npe_batch_size = worth_it_summary.get("npe_parallel_generation_batch_size")
    if flow_batch_size is not None and koopman_batch_size is not None:
        batch_size_text = f"Parallel posterior batch sizes: flow={int(flow_batch_size)}, koopman={int(koopman_batch_size)}"
        if npe_batch_size is not None:
            batch_size_text += f", npe={int(npe_batch_size)}"
        batch_size_text += " samples"
        plt.gcf().text(
            0.98,
            0.02,
            (
                f"Each timing call generates one posterior for one observation.\n"
                f"{batch_size_text}"
            ),
            ha="right",
            va="bottom",
            fontsize=8,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def _plot_worth_it_inference_curve(
    rows: List[Dict[str, float | int]],
    worth_it_summary: Dict[str, float | int | None],
    output_path: Path,
) -> None:
    num_observations = [row["num_observations"] for row in rows]
    series = {
        "flow matching": [row["flow_inference_only_time_seconds"] for row in rows],
        "koopman": [row["koopman_inference_only_time_seconds"] for row in rows],
    }
    if rows and "npe_inference_only_time_seconds" in rows[0]:
        series["npe"] = [row["npe_inference_only_time_seconds"] for row in rows]

    plt.figure(figsize=(8, 5))
    colors = {"flow matching": "tab:blue", "koopman": "tab:orange", "npe": "tab:green"}
    for label, values in series.items():
        plt.plot(num_observations, values, label=label, linewidth=2.0, color=colors.get(label))

    for model_key, row_key, intersection_key, color in [
        (
            "koopman",
            "koopman_inference_only_time_seconds",
            "koopman_inference_only_intersection_num_observations",
            "tab:orange",
        ),
        (
            "npe",
            "npe_inference_only_time_seconds",
            "npe_inference_only_intersection_num_observations",
            "tab:green",
        ),
    ]:
        if model_key == "npe" and "npe" not in series:
            continue
        crossover_count = worth_it_summary.get(intersection_key)
        if crossover_count is None:
            continue
        crossover_x = float(crossover_count)
        crossover_y = None
        for row in rows:
            if int(row["num_observations"]) == int(crossover_count):
                crossover_y = float(row[row_key])
                break
        if crossover_y is None:
            base_seconds = (
                float(worth_it_summary["koopman_teacher_data_time_seconds"])
                if model_key == "koopman"
                else 0.0
            )
            slope_ms = (
                float(worth_it_summary["koopman_mean_sampling_time_per_observation_ms"])
                if model_key == "koopman"
                else float(worth_it_summary["npe_mean_sampling_time_per_observation_ms"])
            )
            crossover_y = base_seconds + (slope_ms / 1000.0) * crossover_x
        plt.scatter([crossover_x], [crossover_y], color=color, s=30, zorder=5)
        plt.annotate(
            f"{model_key}: {int(crossover_count)} obs",
            xy=(crossover_x, crossover_y),
            xytext=(8, 8),
            textcoords="offset points",
            ha="left",
            va="bottom",
            color=color,
        )

    plt.xlabel("Number of independent observations")
    plt.ylabel("Inference-only time (seconds, log scale)")
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Inference-only runtime comparison")
    flow_batch_size = worth_it_summary.get("flow_parallel_generation_batch_size")
    koopman_batch_size = worth_it_summary.get("koopman_parallel_generation_batch_size")
    npe_batch_size = worth_it_summary.get("npe_parallel_generation_batch_size")
    if flow_batch_size is not None and koopman_batch_size is not None:
        batch_size_text = f"Parallel posterior batch sizes: flow={int(flow_batch_size)}, koopman={int(koopman_batch_size)}"
        if npe_batch_size is not None:
            batch_size_text += f", npe={int(npe_batch_size)}"
        batch_size_text += " samples"
        plt.gcf().text(
            0.98,
            0.02,
            (
                "Includes Koopman teacher trajectory generation time, excludes all model training time.\n"
                f"{batch_size_text}"
            ),
            ha="right",
            va="bottom",
            fontsize=8,
        )
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
