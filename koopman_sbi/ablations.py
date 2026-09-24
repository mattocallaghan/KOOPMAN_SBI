from __future__ import annotations

import copy
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import yaml

from koopman_sbi.config import ExperimentConfig, load_experiment_config, save_resolved_config
from koopman_sbi.experiments import run_distill_koopman, run_train_flow


@dataclass
class FlowSimulationAblationConfig:
    base_config: str
    output_root: str
    num_train_samples: List[int]
    save_observation_plots: bool = False


@dataclass
class TeacherGridAblationConfig:
    base_config: str
    output_root: str
    teacher_num_samples: List[int]
    teacher_context_ratios: List[float]
    save_observation_plots: bool = False


def _load_yaml(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"Expected mapping in {path}, got {type(data).__name__}")
    return data


def _resolve_base_config_path(config_path: Path, base_config: str) -> Path:
    candidate = Path(base_config)
    if candidate.is_absolute():
        return candidate
    return (config_path.parent / candidate).resolve()


def _read_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _sanitize_run_name(prefix: str, parts: Dict[str, int]) -> str:
    suffix = "_".join(f"{key}_{value}" for key, value in parts.items())
    return f"{prefix}_{suffix}"


def _prefix_keys(prefix: str, values: Dict[str, Any]) -> Dict[str, Any]:
    return {f"{prefix}_{key}": value for key, value in values.items()}


def _prepare_flow_variant_config(
    base_config: ExperimentConfig,
    output_root: Path,
    num_train_samples: int,
    save_observation_plots: bool,
) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.logging.output_root = str(output_root)
    config.logging.run_name = _sanitize_run_name("flow", {"num_train_samples": num_train_samples})
    config.task.num_train_samples = num_train_samples
    config.task.use_cached_dataset = True
    config.task.dataset_dir = str(output_root / config.task.name / "shared" / "datasets" / f"num_train_samples_{num_train_samples}")
    config.evaluation.save_observation_plots = save_observation_plots
    return config


def _prepare_flow_ablation_koopman_config(
    flow_config: ExperimentConfig,
    flow_checkpoint_path: Path,
    output_root: Path,
    num_train_samples: int,
    save_observation_plots: bool,
) -> ExperimentConfig:
    config = copy.deepcopy(flow_config)
    config.logging.output_root = str(output_root)
    config.logging.run_name = _sanitize_run_name("koopman", {"num_train_samples": num_train_samples})
    config.teacher.checkpoint_path = str(flow_checkpoint_path)
    config.teacher.trajectory_dir = str(
        output_root
        / config.task.name
        / "shared"
        / "teacher_trajectories"
        / f"num_train_samples_{num_train_samples}"
        / f"teacher_num_samples_{config.teacher.num_samples}"
        / f"teacher_num_context_{config.teacher.num_context}"
    )
    config.evaluation.save_observation_plots = save_observation_plots
    return config


def _prepare_teacher_grid_variant_config(
    base_config: ExperimentConfig,
    output_root: Path,
    teacher_num_samples: int,
    teacher_context_ratio: float,
    save_observation_plots: bool,
) -> ExperimentConfig:
    config = copy.deepcopy(base_config)
    config.logging.output_root = str(output_root)
    teacher_num_context = max(1, int(round(teacher_context_ratio * teacher_num_samples)))
    config.logging.run_name = _sanitize_run_name(
        "koopman",
        {
            "teacher_num_samples": teacher_num_samples,
            "teacher_context_ratio_pct": int(round(teacher_context_ratio * 100)),
        },
    )
    config.teacher.num_samples = teacher_num_samples
    config.teacher.num_context = teacher_num_context
    config.teacher.trajectory_dir = str(
        output_root
        / config.task.name
        / "shared"
        / "teacher_trajectories"
        / f"teacher_num_samples_{teacher_num_samples}"
        / f"teacher_context_ratio_{teacher_context_ratio:.3f}"
    )
    config.evaluation.save_observation_plots = save_observation_plots
    return config


def _save_variant_config(config: ExperimentConfig, config_dir: Path, filename: str) -> Path:
    config_path = config_dir / filename
    save_resolved_config(config, str(config_path))
    return config_path


def _plot_flow_ablation(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    metrics = [
        ("mean_c2st", "C2ST"),
        ("mean_mmd", "MMD"),
        ("mean_posterior_mean_error", "Mean Error"),
        ("mean_posterior_variance_ratio", "Variance Ratio"),
        ("training_time_seconds", "Train Time (s)"),
    ]
    x_values = [int(row["num_train_samples"]) for row in rows]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4))
    for axis, (key, label) in zip(np.atleast_1d(axes), metrics):
        flow_key = f"flow_{key}"
        koopman_key = "koopman_training_plus_teacher_data_time_seconds" if key == "training_time_seconds" else f"koopman_{key}"
        flow_values = [float(row[flow_key]) for row in rows]
        koopman_values = [float(row[koopman_key]) for row in rows]
        axis.plot(x_values, flow_values, marker="o", linewidth=2, label="flow_matching")
        axis.plot(x_values, koopman_values, marker="s", linewidth=2, label="koopman")
        axis.set_title(label)
        axis.set_xlabel("num train samples")
        axis.set_xscale("log")
        axis.grid(alpha=0.3)
    handles, labels = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "flow_simulation_ablation.png", dpi=150)
    plt.close(fig)


def _plot_teacher_grid_heatmaps(rows: List[Dict[str, Any]], output_dir: Path) -> None:
    if not rows:
        return
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    sample_values = sorted({int(row["teacher_num_samples"]) for row in rows})
    context_values = sorted({float(row["teacher_context_ratio"]) for row in rows})
    metrics = [
        ("mean_c2st", "C2ST"),
        ("mean_mmd", "MMD"),
        ("mean_posterior_mean_error", "Mean Error"),
        ("mean_posterior_variance_ratio", "Variance Ratio"),
        ("training_plus_teacher_data_time_seconds", "Teacher + Train Time (s)"),
    ]
    fig, axes = plt.subplots(1, len(metrics), figsize=(4 * len(metrics), 4.5))
    for axis, (key, label) in zip(np.atleast_1d(axes), metrics):
        grid = np.full((len(context_values), len(sample_values)), np.nan, dtype=float)
        for row in rows:
            sample_index = sample_values.index(int(row["teacher_num_samples"]))
            context_index = context_values.index(float(row["teacher_context_ratio"]))
            grid[context_index, sample_index] = float(row[key])
        image = axis.imshow(grid, aspect="auto", origin="lower")
        axis.set_title(label)
        axis.set_xlabel("teacher num samples")
        axis.set_ylabel("teacher context ratio")
        axis.set_xticks(range(len(sample_values)))
        axis.set_xticklabels([str(value) for value in sample_values], rotation=45, ha="right")
        axis.set_yticks(range(len(context_values)))
        axis.set_yticklabels([f"{value:.1f}" for value in context_values])
        fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    fig.tight_layout()
    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "teacher_grid_ablation.png", dpi=150)
    plt.close(fig)


def run_flow_simulation_ablation(config_path: str) -> Path:
    ablation_config_path = Path(config_path).resolve()
    raw = _load_yaml(ablation_config_path)
    config = FlowSimulationAblationConfig(**raw)
    base_config_path = _resolve_base_config_path(ablation_config_path, config.base_config)
    base_config = load_experiment_config(str(base_config_path))

    output_root = Path(config.output_root)
    config_dir = output_root / base_config.task.name / "ablation_flow_simulations" / "generated_configs"
    summary_rows: List[Dict[str, Any]] = []

    for num_train_samples in config.num_train_samples:
        flow_config = _prepare_flow_variant_config(
            base_config=base_config,
            output_root=output_root,
            num_train_samples=num_train_samples,
            save_observation_plots=config.save_observation_plots,
        )
        flow_config_path = _save_variant_config(
            flow_config,
            config_dir,
            f"flow_num_train_samples_{num_train_samples}.yaml",
        )
        flow_artifacts = run_train_flow(str(flow_config_path))
        flow_run_summary = _read_json(flow_artifacts.run_paths.base_dir / "run_summary.json")
        flow_evaluation_summary = _read_json(flow_artifacts.run_paths.base_dir / "evaluation" / "flow_matching_summary.json")

        koopman_config = _prepare_flow_ablation_koopman_config(
            flow_config=flow_config,
            flow_checkpoint_path=flow_artifacts.checkpoint_path,
            output_root=output_root,
            num_train_samples=num_train_samples,
            save_observation_plots=config.save_observation_plots,
        )
        koopman_config_path = _save_variant_config(
            koopman_config,
            config_dir,
            f"koopman_num_train_samples_{num_train_samples}.yaml",
        )
        koopman_artifacts = run_distill_koopman(str(koopman_config_path))
        koopman_run_summary = _read_json(koopman_artifacts.run_paths.base_dir / "run_summary.json")
        koopman_evaluation_summary = _read_json(koopman_artifacts.run_paths.base_dir / "evaluation" / "koopman_summary.json")
        summary_rows.append(
            {
                "num_train_samples": num_train_samples,
                "flow_run_name": flow_config.logging.run_name,
                "flow_resolved_config_path": str(flow_config_path),
                "flow_checkpoint_path": str(flow_artifacts.checkpoint_path),
                "koopman_run_name": koopman_config.logging.run_name,
                "koopman_resolved_config_path": str(koopman_config_path),
                "koopman_checkpoint_path": str(koopman_artifacts.checkpoint_path),
                **_prefix_keys("flow", flow_run_summary),
                **_prefix_keys("flow", flow_evaluation_summary),
                **_prefix_keys("koopman", koopman_run_summary),
                **_prefix_keys("koopman", koopman_evaluation_summary),
            }
        )

    summary_rows.sort(key=lambda row: int(row["num_train_samples"]))
    ablation_dir = output_root / base_config.task.name / "ablation_flow_simulations"
    _write_csv(ablation_dir / "summary.csv", summary_rows)
    with open(ablation_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    _plot_flow_ablation(summary_rows, ablation_dir / "plots")
    return ablation_dir


def run_teacher_grid_ablation(config_path: str) -> Path:
    ablation_config_path = Path(config_path).resolve()
    raw = _load_yaml(ablation_config_path)
    config = TeacherGridAblationConfig(**raw)
    base_config_path = _resolve_base_config_path(ablation_config_path, config.base_config)
    base_config = load_experiment_config(str(base_config_path))

    output_root = Path(config.output_root)
    config_dir = output_root / base_config.task.name / "ablation_teacher_grid" / "generated_configs"
    summary_rows: List[Dict[str, Any]] = []

    for teacher_num_samples in config.teacher_num_samples:
        for teacher_context_ratio in config.teacher_context_ratios:
            variant_config = _prepare_teacher_grid_variant_config(
                base_config=base_config,
                output_root=output_root,
                teacher_num_samples=teacher_num_samples,
                teacher_context_ratio=teacher_context_ratio,
                save_observation_plots=config.save_observation_plots,
            )
            variant_config_path = _save_variant_config(
                variant_config,
                config_dir,
                (
                    f"koopman_teacher_num_samples_{teacher_num_samples}"
                    f"_teacher_context_ratio_{teacher_context_ratio:.3f}.yaml"
                ),
            )
            artifacts = run_distill_koopman(str(variant_config_path))
            run_summary = _read_json(artifacts.run_paths.base_dir / "run_summary.json")
            evaluation_summary = _read_json(artifacts.run_paths.base_dir / "evaluation" / "koopman_summary.json")
            summary_rows.append(
                {
                    "teacher_num_samples": teacher_num_samples,
                    "teacher_num_context": int(variant_config.teacher.num_context),
                    "teacher_context_ratio": float(teacher_context_ratio),
                    "run_name": variant_config.logging.run_name,
                    "resolved_config_path": str(variant_config_path),
                    "checkpoint_path": str(artifacts.checkpoint_path),
                    **run_summary,
                    **evaluation_summary,
                }
            )

    summary_rows.sort(key=lambda row: (int(row["teacher_num_samples"]), float(row["teacher_context_ratio"])))
    ablation_dir = output_root / base_config.task.name / "ablation_teacher_grid"
    _write_csv(ablation_dir / "summary.csv", summary_rows)
    with open(ablation_dir / "summary.json", "w", encoding="utf-8") as handle:
        json.dump(summary_rows, handle, indent=2)
    _plot_teacher_grid_heatmaps(summary_rows, ablation_dir / "plots")
    return ablation_dir
