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
from koopman_sbi.tasks import get_task
import torch

from koopman_sbi.config import ExperimentConfig, resolve_task_config_path, save_resolved_config
from koopman_sbi.data import DatasetBundle, load_or_generate_dataset
from koopman_sbi.experiments.pipeline import (
    _load_config,
    _read_run_summary_from_checkpoint,
    _resolve_teacher_checkpoint,
    run_distill_koopman,
)
from koopman_sbi.models import ConditionalFlowMatching, KoopmanFlow
from koopman_sbi.paths import prepare_run_directories
from koopman_sbi.runtime import detect_device, move_tensor_to_device, set_global_seed


DEFAULT_SAMPLE_COUNTS = [
    1,
    10,
    100,
    1_000,
    10_000,
    500_000,
    800_000,
    1_000_000,
    2_000_000,
    5_000_000,
    10_000_000,
    40_000_000,
    100_000_000,
]
DEFAULT_NUM_REPEATS = 3
DEFAULT_WARMUP_SAMPLES = 1
DEFAULT_MAX_CONTEXT_BATCH_SIZE = 1_000_000
CHUNKED_SAMPLE_THRESHOLD = 1_000_000


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

    return models, timing_metadata


def _build_standardized_observation_bank(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
) -> torch.Tensor:
    task = get_task(config.task.name)
    observations = []
    for obs in config.evaluation.observations:
        observation = task.get_observation(num_observation=obs).float()
        if observation.dim() > 1:
            observation = observation.reshape(-1, observation.shape[-1])[0]
        observations.append(observation)
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


def _make_single_observation_context_batch(
    standardized_observation_bank: torch.Tensor,
    observation_index: int,
    num_samples: int,
    device: torch.device,
) -> torch.Tensor:
    bank_size = int(standardized_observation_bank.shape[0])
    observation = standardized_observation_bank[int(observation_index) % bank_size]
    context = observation.unsqueeze(0).repeat(int(num_samples), 1)
    return move_tensor_to_device(context, device)


def _time_model_for_sample_count(
    model,
    standardized_observation_bank: torch.Tensor,
    num_samples: int,
    max_context_batch_size: int,
    num_repeats: int,
) -> Dict[str, float | int]:
    total_requested_samples = int(num_samples)
    if total_requested_samples <= CHUNKED_SAMPLE_THRESHOLD and total_requested_samples > int(max_context_batch_size):
        raise ValueError(
            "Requested full parallel context batch exceeds max_context_batch_size: "
            f"{total_requested_samples} > {int(max_context_batch_size)}"
        )
    effective_chunk_size = (
        total_requested_samples if total_requested_samples <= CHUNKED_SAMPLE_THRESHOLD else CHUNKED_SAMPLE_THRESHOLD
    )
    if effective_chunk_size <= 0:
        raise ValueError("Effective chunk size must be positive.")
    num_sample_chunks = math.ceil(total_requested_samples / effective_chunk_size)

    run_times_ms: List[float] = []
    for _ in range(num_repeats):
        _synchronize_device(model.device)
        start_time = time.perf_counter()
        processed_samples = 0
        while processed_samples < total_requested_samples:
            current_chunk_size = min(effective_chunk_size, total_requested_samples - processed_samples)
            context = _make_single_observation_context_batch(
                standardized_observation_bank=standardized_observation_bank,
                observation_index=0,
                num_samples=current_chunk_size,
                device=model.device,
            )
            with torch.no_grad():
                samples = model.sample_batch(context)
            del context
            del samples
            processed_samples += current_chunk_size
        _synchronize_device(model.device)
        run_times_ms.append((time.perf_counter() - start_time) * 1000.0)

    mean_time_ms = sum(run_times_ms) / len(run_times_ms)
    std_time_ms = math.sqrt(sum((value - mean_time_ms) ** 2 for value in run_times_ms) / len(run_times_ms))
    return {
        "num_observations": 1,
        "num_posterior_samples": int(num_samples),
        "num_samples": int(num_samples),
        "total_requested_samples": total_requested_samples,
        "context_batch_size": int(effective_chunk_size),
        "num_sample_chunks": int(num_sample_chunks),
        "max_context_batch_size": int(max_context_batch_size),
        "num_repeats": int(num_repeats),
        "mean_wall_clock_ms": float(mean_time_ms),
        "std_wall_clock_ms": float(std_time_ms),
        "mean_wall_clock_seconds": float(mean_time_ms / 1000.0),
        "seconds_per_observation": float(mean_time_ms / 1000.0),
        "samples_per_second": float(total_requested_samples / max(mean_time_ms / 1000.0, 1e-12)),
        "observations_per_second": float(1.0 / max(mean_time_ms / 1000.0, 1e-12)),
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
    requested_max_context_batch_size: int,
) -> int:
    candidate = max(1, int(requested_max_context_batch_size))
    minimum_candidate = 1

    while candidate >= minimum_candidate:
        print(f"[gpu-evaluation] auto-tuning context batch size: trying {candidate}")
        all_models_fit = True
        for model_name, model in models.items():
            fits = _can_run_context_batch(
                model=model,
                standardized_observation_bank=standardized_observation_bank,
                num_observations=1,
                num_posterior_samples=candidate,
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
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _append_skipped_rows(
    results: List[Dict[str, float | int | str]],
    *,
    model_name: str,
    sample_counts: List[int],
    max_context_batch_size: int,
    num_repeats: int,
    skip_reason: str,
) -> None:
    for num_samples in sample_counts:
        results.append(
            {
                "model_name": model_name,
                "num_observations": 1,
                "num_posterior_samples": int(num_samples),
                "num_samples": int(num_samples),
                "total_requested_samples": int(num_samples),
                "context_batch_size": min(int(num_samples), CHUNKED_SAMPLE_THRESHOLD),
                "max_context_batch_size": int(max_context_batch_size),
                "num_repeats": int(num_repeats),
                "status": "skipped",
                "skip_reason": skip_reason,
                "flow_reference_100k_seconds": _flow_reference_time_seconds(results),
            }
        )


def _flow_reference_time_seconds(rows: List[Dict[str, float | int | str]], reference_num_samples: int = 100_000) -> float:
    for row in rows:
        if (
            row.get("model_name") == "flow_matching"
            and row.get("status", "ok") == "ok"
            and int(row.get("num_samples", -1)) == int(reference_num_samples)
        ):
            return float(row["mean_wall_clock_seconds"])
    return 0.0


def _plot_gpu_evaluation(
    rows: List[Dict[str, float | int | str]],
    output_path: Path,
    *,
    x_scale: str = "log",
    y_scale: str = "log",
) -> None:
    if not rows:
        return
    flow_reference_seconds = _flow_reference_time_seconds(rows)
    plt.figure(figsize=(8, 5))
    colors = {"flow_matching": "tab:blue", "koopman": "tab:orange"}
    for model_name in sorted({str(row["model_name"]) for row in rows}):
        model_rows = [row for row in rows if row["model_name"] == model_name and row.get("status", "ok") == "ok"]
        if not model_rows:
            continue
        model_rows.sort(key=lambda row: int(row["num_samples"]))
        x_values = [int(row["num_samples"]) for row in model_rows]
        y_values = [float(row.get("plot_wall_clock_seconds", row["mean_wall_clock_seconds"])) for row in model_rows]
        plt.plot(x_values, y_values, marker="o", linewidth=2.0, label=model_name, color=colors.get(model_name))
    plt.xscale(x_scale)
    plt.yscale(y_scale)
    plt.xlabel("Number of samples for one observation")
    plt.ylabel("Wall clock time (seconds)")
    if flow_reference_seconds > 0.0:
        plt.title(
            "GPU posterior sampling wall clock (single observation)\n"
            f"Koopman includes fixed flow_matching@100k offset = {flow_reference_seconds:.3f}s"
        )
    else:
        plt.title("GPU posterior sampling wall clock (single observation)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def run_gpu_evaluation(
    config_path: str,
    num_posterior_samples: int = 1,
    observation_counts: Iterable[int] = DEFAULT_SAMPLE_COUNTS,
    max_context_batch_size: int = DEFAULT_MAX_CONTEXT_BATCH_SIZE,
    num_repeats: int = DEFAULT_NUM_REPEATS,
    warmup_observations: int = DEFAULT_WARMUP_SAMPLES,
    auto_max_context_batch_size: bool = False,
    x_scale: str = "log",
    y_scale: str = "log",
) -> Path:
    config = _load_config(config_path)
    sample_counts = [int(value) for value in observation_counts]
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    models, timing_metadata = _load_models_for_gpu_evaluation(config, config_path, device)
    standardized_observation_bank = _build_standardized_observation_bank(config, dataset_bundle)
    max_context_batch_size = CHUNKED_SAMPLE_THRESHOLD

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
        "num_posterior_samples": 1,
        "sample_counts": sample_counts,
        "max_context_batch_size": int(max_context_batch_size),
        "parallelization_mode": "single_observation_full_batch_over_samples",
        "auto_max_context_batch_size": False,
        "num_repeats": int(num_repeats),
        "warmup_samples": int(warmup_observations),
        "x_scale": str(x_scale),
        "y_scale": str(y_scale),
        "timing_metadata": timing_metadata,
    }
    with open(run_paths.base_dir / "run_summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    results: List[Dict[str, float | int | str]] = []
    csv_path = run_paths.metrics_dir / "gpu_wall_clock.csv"
    plot_path = run_paths.plots_dir / "gpu_wall_clock.png"

    for model_name, model in models.items():
        try:
            warmup_context = _make_single_observation_context_batch(
                standardized_observation_bank=standardized_observation_bank,
                observation_index=0,
                num_samples=warmup_observations,
                device=model.device,
            )
            with torch.no_grad():
                warmup_samples = model.sample_batch(warmup_context)
            del warmup_context
            del warmup_samples
            _synchronize_device(model.device)
        except ImportError as error:
            print(
                f"[gpu-evaluation] warning: skipping model={model_name} because warmup failed: {error}"
            )
            _append_skipped_rows(
                results,
                model_name=model_name,
                sample_counts=sample_counts,
                max_context_batch_size=int(max_context_batch_size),
                num_repeats=int(num_repeats),
                skip_reason="missing_dependency",
            )
            _write_rows(csv_path, results)
            _plot_gpu_evaluation(results, plot_path, x_scale=x_scale, y_scale=y_scale)
            continue

        for num_samples in sample_counts:
            total_requested_samples = int(num_samples)
            print(
                f"[gpu-evaluation] model={model_name} num_samples={int(num_samples)} "
                f"total_contexts={total_requested_samples}"
            )
            if total_requested_samples <= CHUNKED_SAMPLE_THRESHOLD and total_requested_samples > int(max_context_batch_size):
                print(
                    f"[gpu-evaluation] skipping model={model_name} num_samples={int(num_samples)} "
                    f"because full parallel batch {total_requested_samples} exceeds "
                    f"max_context_batch_size={int(max_context_batch_size)}"
                )
                results.append(
                    {
                        "model_name": model_name,
                        "num_observations": 1,
                        "num_posterior_samples": int(num_samples),
                        "num_samples": int(num_samples),
                        "total_requested_samples": total_requested_samples,
                        "context_batch_size": total_requested_samples,
                        "max_context_batch_size": int(max_context_batch_size),
                        "num_repeats": int(num_repeats),
                        "flow_reference_100k_seconds": _flow_reference_time_seconds(results),
                        "status": "skipped",
                        "skip_reason": "exceeds_max_context_batch_size",
                    }
                )
                _write_rows(csv_path, results)
                _plot_gpu_evaluation(results, plot_path, x_scale=x_scale, y_scale=y_scale)
                continue
            try:
                row = _time_model_for_sample_count(
                    model=model,
                    standardized_observation_bank=standardized_observation_bank,
                    num_samples=int(num_samples),
                    max_context_batch_size=int(max_context_batch_size),
                    num_repeats=int(num_repeats),
                )
            except ImportError as error:
                print(
                    f"[gpu-evaluation] warning: skipping remaining points for model={model_name} "
                    f"because sampling requires a missing dependency: {error}"
                )
                remaining_counts = [value for value in sample_counts if int(value) >= int(num_samples)]
                _append_skipped_rows(
                    results,
                    model_name=model_name,
                    sample_counts=remaining_counts,
                    max_context_batch_size=int(max_context_batch_size),
                    num_repeats=int(num_repeats),
                    skip_reason="missing_dependency",
                )
                _write_rows(csv_path, results)
                _plot_gpu_evaluation(results, plot_path, x_scale=x_scale, y_scale=y_scale)
                break
            except RuntimeError as error:
                _clear_device_cache(model.device)
                if _is_out_of_memory_error(error):
                    print(
                        f"[gpu-evaluation] OOM for model={model_name} num_samples={int(num_samples)}; "
                        "marking point as skipped"
                    )
                    results.append(
                        {
                            "model_name": model_name,
                            "num_observations": 1,
                            "num_posterior_samples": int(num_samples),
                            "num_samples": int(num_samples),
                            "total_requested_samples": total_requested_samples,
                            "context_batch_size": total_requested_samples,
                            "max_context_batch_size": int(max_context_batch_size),
                            "num_repeats": int(num_repeats),
                            "flow_reference_100k_seconds": _flow_reference_time_seconds(results),
                            "status": "skipped",
                            "skip_reason": "out_of_memory",
                        }
                    )
                    _write_rows(csv_path, results)
                    _plot_gpu_evaluation(results, plot_path, x_scale=x_scale, y_scale=y_scale)
                    continue
                raise
            flow_reference_100k_seconds = _flow_reference_time_seconds(results)
            row["model_name"] = model_name
            row["flow_reference_100k_seconds"] = flow_reference_100k_seconds
            if model_name == "koopman":
                row["plot_wall_clock_seconds"] = float(row["mean_wall_clock_seconds"]) + flow_reference_100k_seconds
                row["plot_wall_clock_ms"] = float(row["mean_wall_clock_ms"]) + flow_reference_100k_seconds * 1000.0
            else:
                row["plot_wall_clock_seconds"] = float(row["mean_wall_clock_seconds"])
                row["plot_wall_clock_ms"] = float(row["mean_wall_clock_ms"])
            row["status"] = "ok"
            results.append(row)
            _write_rows(csv_path, results)
            _plot_gpu_evaluation(results, plot_path, x_scale=x_scale, y_scale=y_scale)

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
    parser = argparse.ArgumentParser(
        description="Measure full-batch GPU posterior sampling speed for one observation as the number of samples increases."
    )
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Path to experiment YAML config.")
    config_group.add_argument("--task", help="Benchmark task name, resolved via the task config directory.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional directory containing per-task YAML configs. Defaults to `koopman_sbi/configs/tasks`.",
    )
    parser.add_argument(
        "--observation-counts",
        type=int,
        nargs="+",
        default=DEFAULT_SAMPLE_COUNTS,
        help="Sample counts to benchmark for a single observation.",
    )
    parser.add_argument("--max-context-batch-size", type=int, default=DEFAULT_MAX_CONTEXT_BATCH_SIZE)
    parser.add_argument(
        "--auto-max-context-batch-size",
        action="store_true",
        help="Auto-tune the largest feasible context batch size via OOM backoff before benchmarking.",
    )
    parser.add_argument("--num-repeats", type=int, default=DEFAULT_NUM_REPEATS)
    parser.add_argument("--warmup-observations", type=int, default=DEFAULT_WARMUP_SAMPLES)
    parser.add_argument("--x-scale", choices=["log", "linear"], default="log")
    parser.add_argument("--y-scale", choices=["log", "linear"], default="log")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = _resolve_config_argument(args)
    run_dir = run_gpu_evaluation(
        config_path=config_path,
        num_posterior_samples=1,
        observation_counts=args.observation_counts,
        max_context_batch_size=args.max_context_batch_size,
        num_repeats=args.num_repeats,
        warmup_observations=args.warmup_observations,
        auto_max_context_batch_size=args.auto_max_context_batch_size,
        x_scale=args.x_scale,
        y_scale=args.y_scale,
    )
    print(f"Saved GPU evaluation artifacts to {run_dir}")


if __name__ == "__main__":
    main()
