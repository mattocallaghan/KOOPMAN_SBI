from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from koopman_sbi.runtime import timestamped_run_name


@dataclass
class RunPaths:
    base_dir: Path
    checkpoints_dir: Path
    plots_dir: Path
    metrics_dir: Path
    last_model_dir: Path


def prepare_run_directories(
    output_root: str,
    task_name: str,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> RunPaths:
    run_id = run_name or timestamped_run_name(experiment_name)
    base_dir = Path(output_root) / task_name / experiment_name / run_id
    checkpoints_dir = base_dir / "checkpoints"
    plots_dir = base_dir / "plots"
    metrics_dir = base_dir / "metrics"
    last_model_dir = Path(output_root) / task_name / "last_model" / experiment_name
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    last_model_dir.mkdir(parents=True, exist_ok=True)
    return RunPaths(
        base_dir=base_dir,
        checkpoints_dir=checkpoints_dir,
        plots_dir=plots_dir,
        metrics_dir=metrics_dir,
        last_model_dir=last_model_dir,
    )


def resolve_dataset_dir(output_root: str, task_name: str, configured_dir: Optional[str]) -> Path:
    if configured_dir:
        return Path(configured_dir)
    return Path(output_root) / task_name / "shared" / "dataset"


def resolve_teacher_dir(output_root: str, task_name: str, configured_dir: Optional[str]) -> Path:
    if configured_dir:
        return Path(configured_dir)
    return Path(output_root) / task_name / "shared" / "teacher_trajectories"
