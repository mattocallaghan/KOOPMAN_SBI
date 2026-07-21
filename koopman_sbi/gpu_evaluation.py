from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import sbibm
import torch

from koopman_sbi.config import ExperimentConfig, resolve_task_config_path, save_resolved_config
from koopman_sbi.data import DatasetBundle, load_or_generate_dataset
from koopman_sbi.experiments.pipeline import (
    _load_config,
    _read_run_summary_from_checkpoint,
    _resolve_npe_checkpoint,
    _resolve_teacher_checkpoint,
    run_distill_koopman,
)
from koopman_sbi.models import ConditionalFlowMatching, KoopmanFlow, NormalizingFlowNPE
from koopman_sbi.paths import prepare_run_directories
from koopman_sbi.runtime import detect_device, move_tensor_to_device, set_global_seed


DEFAULT_OBSERVATION_COUNTS = [1, 10, 100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]
DEFAULT_NUM_POSTERIOR_SAMPLES = 1_000
DEFAULT_NUM_REPEATS = 3
DEFAULT_WARMUP_OBSERVATIONS = 1
DEFAULT_MAX_CONTEXT_BATCH_SIZE = 100_000


def _synchronize_device(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.synchronize(device)
    elif device.type == "mps" and torch.backends.mps.is_available():
        torch.mps.synchronize()


def _clear_device_cache(device: torch.device) -> None:
    if device.type == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif device.type == "mps" and torch.backends.mps.is_available() and hasattr(torch.mps, "empty_cache"):
        torch.mps.empty_cache()


def _is_out_of_memory_error(error: RuntimeError) -> bool:
    message = str(error).lower()
    return (
        "out of memory" in message
        or "cuda out of memory" in message
        or "mps backend out of memory" in message
    )


def _resolve_config_argument(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config
    return str(resolve_task_config_path(args.task, args.config_dir))


def _resolve_koopman_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.evaluation.koopman_checkpoint_path and Path(config.evaluation.koopman_checkpoint_path).exists():
        return Path(config.evaluation.koopman_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "distill_koopman"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "distill_koopman"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    return run_distill_koopman(config_path).checkpoint_path


def _load_models_for_gpu_evaluation(
    config: ExperimentConfig,
    config_path: str,
    device: torch.device,
) -> tuple[Dict[str, object], Dict[str, Dict[str, float | int | bool | str]]]:
    flow_checkpoint = (
        Path(config.evaluation.flow_checkpoint_path)
        if config.evaluation.flow_checkpoint_path and Path(config.evaluation.flow_checkpoint_path).exists()
        else _resolve_teacher_checkpoint(config, config_path)
    )
    koopman_checkpoint = _resolve_koopman_checkpoint(config, config_path)

    models: Dict[str, object] = {
        "flow_matching": ConditionalFlowMatching.load(str(flow_checkpoint), device=device),
        "koopman": KoopmanFlow.load(str(koopman_checkpoint), device=device),
    }
    timing_metadata: Dict[str, Dict[str, float | int | bool | str]] = {
        "flow_matching": _read_run_summary_from_checkpoint(flow_checkpoint),
        "koopman": _read_run_summary_from_checkpoint(koopman_checkpoint),
    }

    if config.evaluation.include_npe or config.evaluation.npe_checkpoint_path:
        try:
            npe_checkpoint = _resolve_npe_checkpoint(config, config_path)
        except FileNotFoundError:
            npe_checkpoint = None
        if npe_checkpoint is not None:
            models["npe"] = NormalizingFlowNPE.load(str(npe_checkpoint), device=device)
            timing_metadata["npe"] = _read_run_summary_from_checkpoint(npe_checkpoint)

    return models, timing_metadata


def _build_standardized_observation_bank(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
) -> torch.Tensor:
    task = sbibm.get_task(config.task.name)
    observations = [task.get_observation(num_observation=obs).float() for obs in config.evaluation.observations]
    stacked = torch.stack(observations, dim=0)
    return dataset_bundle.standardizer.standardize_x(stacked)


def _make_context_batch(
    standardized_observation_bank: torch.Tensor,
    start_index: int,
    num_observations: int,
    num_posterior_samples: int,
    device: torch.device,
) -> torch.Tensor:
    bank_size = int(standardized_observation_bank.shape[0])
    indices = (torch.arange(start_index, start_index + num_observations) % bank_size).long()
    observation_batch = standardized_observation_bank[indices]
    context = observation_batch.repeat_interleave(num_posterior_samples, dim=0)
    return move_tensor_to_device(context, device)


def _time_model_for_observation_count(
    model,
    standardized_observation_bank: torch.Tensor,
    num_observations: int,
    num_posterior_samples: int,
    max_context_batch_size: int,
    num_repeats: int,
) -> Dict[str, float | int]:
    observation_chunk_size = max(1, max_context_batch_size // max(num_posterior_samples, 1))
    total_requested_samples = int(num_observations * num_posterior_samples)

    run_times_ms: List[float] = []
    for _ in range(num_repeats):
        _synchronize_device(model.device)
        start_time = time.perf_counter()
        num_processed_observations = 0
        while num_processed_observations < num_observations:
            chunk_observations = min(observation_chunk_size, num_observations - num_processed_observations)
            context = _make_context_batch(
                standardized_observation_bank=standardized_observation_bank,
                start_index=num_processed_observations,
                num_observations=chunk_observations,
                num_posterior_samples=num_posterior_samples,
                device=model.device,
            )
            with torch.no_grad():
                samples = model.sample_batch(context)
            del context
            del samples
            num_processed_observations += chunk_observations
        _synchronize_device(model.device)
        run_times_ms.append((time.perf_counter() - start_time) * 1000.0)

    mean_time_ms = sum(run_times_ms) / len(run_times_ms)
    std_time_ms = math.sqrt(sum((value - mean_time_ms) ** 2 for value in run_times_ms) / len(run_times_ms))
    return {
        "num_observations": int(num_observations),
        "num_posterior_samples": int(num_posterior_samples),
        "total_requested_samples": total_requested_samples,
        "observation_chunk_size": int(observation_chunk_size),
        "max_context_batch_size": int(max_context_batch_size),
        "num_repeats": int(num_repeats),
        "mean_wall_clock_ms": float(mean_time_ms),
        "std_wall_clock_ms": float(std_time_ms),
        "mean_wall_clock_seconds": float(mean_time_ms / 1000.0),
        "seconds_per_observation": float((mean_time_ms / 1000.0) / max(num_observations, 1)),
        "samples_per_second": float(total_requested_samples / max(mean_time_ms / 1000.0, 1e-12)),
        "observations_per_second": float(num_observations / max(mean_time_ms / 1000.0, 1e-12)),
    }


def _can_run_context_batch(
    model,
    standardized_observation_bank: torch.Tensor,
    num_observations: int,
    num_posterior_samples: int,
) -> bool:
    try:
        context = _make_context_batch(
            standardized_observation_bank=standardized_observation_bank,
            start_index=0,
            num_observations=num_observations,
            num_posterior_samples=num_posterior_samples,
            device=model.device,
        )
        with torch.no_grad():
            samples = model.sample_batch(context)
        del context
        del samples
        _synchronize_device(model.device)
        _clear_device_cache(model.device)
        return True
    except RuntimeError as error:
        _clear_device_cache(model.device)
        if _is_out_of_memory_error(error):
            return False
        raise


def _auto_tune_max_context_batch_size(
    models: Dict[str, object],
    standardized_observation_bank: torch.Tensor,
    num_posterior_samples: int,
    requested_max_context_batch_size: int,
) -> int:
    candidate = max(int(num_posterior_samples), int(requested_max_context_batch_size))
    minimum_candidate = max(int(num_posterior_samples), 1)

    while candidate >= minimum_candidate:
        print(f"[gpu-evaluation] auto-tuning context batch size: trying {candidate}")
        all_models_fit = True
        candidate_observations = max(1, candidate // max(num_posterior_samples, 1))
        for model_name, model in models.items():
            fits = _can_run_context_batch(
                model=model,
                standardized_observation_bank=standardized_observation_bank,
                num_observations=candidate_observations,
                num_posterior_samples=num_posterior_samples,
            )
            if not fits:
                print(f"[gpu-evaluation] auto-tuning: {model_name} OOM at context batch size {candidate}")
                all_models_fit = False
                break
        if all_models_fit:
            print(f"[gpu-evaluation] auto-tuning selected context batch size {candidate}")
            return candidate
        if candidate == minimum_candidate:
            break
        candidate = max(minimum_candidate, candidate // 2)

    raise RuntimeError(
        "Failed to find a feasible context batch size during auto-tuning. "
        "Try reducing `--num-posterior-samples`."
    )


def _write_rows(path: Path, rows: List[Dict[str, float | int | str]]) -> None:
    if not rows:
        return
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot_gpu_evaluation(rows: List[Dict[str, float | int | str]], output_path: Path) -> None:
    if not rows:
        return
    plt.figure(figsize=(8, 5))
    colors = {"flow_matching": "tab:blue", "koopman": "tab:orange", "npe": "tab:green"}
    for model_name in sorted({str(row["model_name"]) for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name]
        model_rows.sort(key=lambda row: int(row["num_observations"]))
        x_values = [int(row["num_observations"]) for row in model_rows]
        y_values = [float(row["mean_wall_clock_seconds"]) for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=2.0, label=model_name, color=colors.get(model_name))
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Number of independent observations")
    plt.ylabel("Wall clock time (seconds)")
    plt.title("GPU posterior sampling wall clock")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_gpu_evaluation(
    config_path: str,
    num_posterior_samples: int = DEFAULT_NUM_POSTERIOR_SAMPLES,
    observation_counts: Iterable[int] = DEFAULT_OBSERVATION_COUNTS,
    max_context_batch_size: int = DEFAULT_MAX_CONTEXT_BATCH_SIZE,
    num_repeats: int = DEFAULT_NUM_REPEATS,
    warmup_observations: int = DEFAULT_WARMUP_OBSERVATIONS,
    auto_max_context_batch_size: bool = False,
) -> Path:
    config = _load_config(config_path)
    observation_counts = [int(value) for value in observation_counts]
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    models, timing_metadata = _load_models_for_gpu_evaluation(config, config_path, device)
    standardized_observation_bank = _build_standardized_observation_bank(config, dataset_bundle)
    if auto_max_context_batch_size:
        max_context_batch_size = _auto_tune_max_context_batch_size(
            models=models,
            standardized_observation_bank=standardized_observation_bank,
            num_posterior_samples=int(num_posterior_samples),
            requested_max_context_batch_size=int(max_context_batch_size),
        )

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="gpu_evaluation",
        run_name=config.logging.run_name,
    )
    save_resolved_config(config, str(run_paths.base_dir / "resolved_config.yaml"))

    summary = {
        "experiment_name": "gpu_evaluation",
        "device": str(device),
        "num_posterior_samples": int(num_posterior_samples),
        "observation_counts": observation_counts,
        "max_context_batch_size": int(max_context_batch_size),
        "auto_max_context_batch_size": bool(auto_max_context_batch_size),
        "num_repeats": int(num_repeats),
        "warmup_observations": int(warmup_observations),
        "timing_metadata": timing_metadata,
    }
    with open(run_paths.base_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    results: List[Dict[str, float | int | str]] = []
    csv_path = run_paths.metrics_dir / "gpu_wall_clock.csv"
    plot_path = run_paths.plots_dir / "gpu_wall_clock.png"

    for model_name, model in models.items():
        warmup_context = _make_context_batch(
            standardized_observation_bank=standardized_observation_bank,
            start_index=0,
            num_observations=warmup_observations,
            num_posterior_samples=num_posterior_samples,
            device=model.device,
        )
        with torch.no_grad():
            warmup_samples = model.sample_batch(warmup_context)
        del warmup_context
        del warmup_samples
        _synchronize_device(model.device)

        for num_observations in observation_counts:
            print(
                f"[gpu-evaluation] model={model_name} observations={int(num_observations)} "
                f"samples_per_posterior={num_posterior_samples}"
            )
            row = _time_model_for_observation_count(
                model=model,
                standardized_observation_bank=standardized_observation_bank,
                num_observations=int(num_observations),
                num_posterior_samples=int(num_posterior_samples),
                max_context_batch_size=int(max_context_batch_size),
                num_repeats=int(num_repeats),
            )
            row["model_name"] = model_name
            results.append(row)
            _write_rows(csv_path, results)
            _plot_gpu_evaluation(results, plot_path)

    with open(run_paths.metrics_dir / "gpu_wall_clock_summary.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                **summary,
                "results": results,
                "plot_path": str(plot_path),
                "csv_path": str(csv_path),
            },
            handle,
            indent=2,
        )
    return run_paths.base_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure wall clock posterior sampling speed across many observations.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Path to experiment YAML config.")
    config_group.add_argument("--task", help="Benchmark task name, resolved via the task config directory.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional directory containing per-task YAML configs. Defaults to `koopman_sbi/configs/tasks`.",
    )
    parser.add_argument("--num-posterior-samples", type=int, default=DEFAULT_NUM_POSTERIOR_SAMPLES)
    parser.add_argument(
        "--observation-counts",
        type=int,
        nargs="+",
        default=DEFAULT_OBSERVATION_COUNTS,
        help="Observation counts M to benchmark.",
    )
    parser.add_argument("--max-context-batch-size", type=int, default=DEFAULT_MAX_CONTEXT_BATCH_SIZE)
    parser.add_argument(
        "--auto-max-context-batch-size",
        action="store_true",
        help="Auto-tune the largest feasible context batch size via OOM backoff before benchmarking.",
    )
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--warmup-observations", type=int, default=DEFAULT_WARMUP_OBSERVATIONS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = _resolve_config_argument(args)
    run_dir = run_gpu_evaluation(
        config_path=config_path,
        num_posterior_samples=args.num_posterior_samples,
        observation_counts=args.observation_counts,
        max_context_batch_size=args.max_context_batch_size,
        num_repeats=args.num_repeats,
        warmup_observations=args.warmup_observations,
        auto_max_context_batch_size=args.auto_max_context_batch_size,
    )
    print(f"Saved GPU evaluation artifacts to {run_dir}")


if __name__ == "__main__":
    main()
