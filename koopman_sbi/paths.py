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


def resolve_existing_run_dir(
    output_root: str,
    task_name: str,
    experiment_name: str,
    run_name: Optional[str] = None,
) -> Path:
    experiment_dir = Path(output_root) / task_name / experiment_name
    if run_name:
        run_dir = experiment_dir / run_name
        if not run_dir.exists():
            raise FileNotFoundError(f"Run directory does not exist: {run_dir}")
        return run_dir
    if not experiment_dir.exists():
        raise FileNotFoundError(f"Experiment directory does not exist: {experiment_dir}")
    candidate_dirs = [path for path in experiment_dir.iterdir() if path.is_dir()]
    if not candidate_dirs:
        raise FileNotFoundError(f"No runs found under {experiment_dir}")
    return max(candidate_dirs, key=lambda path: path.stat().st_mtime)
