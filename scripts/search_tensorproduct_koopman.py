#!/usr/bin/env python
"""Adaptive Optuna search for tensor-product Koopman posterior recovery."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cache_dir = ROOT / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("KERAS_BACKEND", "torch")

try:
    import optuna
except ImportError as error:
    raise SystemExit("Install Optuna in this environment: python -m pip install optuna") from error

from koopman_sbi.config import load_experiment_config, save_resolved_config
from koopman_sbi.experiments import run_train_tensorproduct_koopman


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("Search config must be a YAML mapping.")
    return value


def positive_choices(raw: Any, name: str) -> list[int]:
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{name} must be a non-empty list.")
    values = [int(value) for value in raw]
    if min(values) < 1:
        raise ValueError(f"{name} values must be positive.")
    return values


def suggest_weight(trial: optuna.Trial, name: str, spec: Any) -> float:
    if isinstance(spec, list):
        return float(trial.suggest_categorical(name, [float(value) for value in spec]))
    if isinstance(spec, dict):
        low, high = float(spec["low"]), float(spec["high"])
        if low <= 0 or high < low:
            raise ValueError(f"Invalid {name} range.")
        return float(trial.suggest_float(name, low, high, log=bool(spec.get("log", False))))
    return float(spec)


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with path.with_suffix(".json").open("w", encoding="utf-8") as handle:
        json.dump(rows, handle, indent=2)


def export_study(task_name: str, study: optuna.Study, output_root: Path) -> None:
    rows: list[dict[str, Any]] = []
    pareto_trials = {trial.number for trial in study.best_trials}
    for frozen in study.trials:
        values = frozen.values or []
        rows.append(
            {
                "task": task_name,
                "trial_number": frozen.number,
                "status": frozen.state.name.lower(),
                "objective_mean_c2st": values[0] if len(values) > 0 else None,
                "objective_sampling_time_per_generated_sample_ms": values[1] if len(values) > 1 else None,
                "pareto_optimal": frozen.number in pareto_trials,
                **frozen.params,
                **frozen.user_attrs,
            }
        )
    rows.sort(
        key=lambda row: (
            row["status"] != "complete",
            row["objective_mean_c2st"] is None,
            row["objective_mean_c2st"] or float("inf"),
        )
    )
    write_rows(output_root / f"{task_name}_posterior_recovery_trials.csv", rows)


def make_objective(task_name: str, task_config: Path, raw: dict[str, Any], output_root: Path):
    contexts = positive_choices(raw["num_context"], "num_context")
    dimensions = positive_choices(raw.get("lifting_dim", raw.get("latent_dim")), "lifting_dim")
    ranks = positive_choices(raw["tensor_rank"], "tensor_rank")
    weights = raw.get("loss_weights")
    if not isinstance(weights, dict):
        raise ValueError("loss_weights must be a mapping.")
    evaluation_samples = raw.get("evaluation_num_posterior_samples")
    if evaluation_samples is not None:
        evaluation_samples = int(evaluation_samples)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        num_context = trial.suggest_categorical("num_context", contexts)
        lifting_dim = trial.suggest_categorical("lifting_dim", dimensions)
        tensor_rank = trial.suggest_categorical("tensor_rank", ranks)
        lambda_ae = suggest_weight(trial, "lambda_ae", weights["lambda_ae"])
        lambda_lat = suggest_weight(trial, "lambda_lat", weights["lambda_lat"])
        lambda_end = suggest_weight(trial, "lambda_end", weights["lambda_end"])
        name = f"optuna-{trial.number:04d}"

        variant = copy.deepcopy(load_experiment_config(str(task_config)))
        variant.task.name = task_name
        # Each trial uses the existing flow checkpoint. It may generate cached
        # teacher trajectories, but is never allowed to retrain the teacher.
        variant.teacher.auto_train_if_missing = False
        variant.teacher.generate_trajectories = True
        variant.benchmark_suite.train_flow_matching = False
        variant.teacher.num_context = num_context
        variant.teacher.trajectory_dir = str(output_root / "teacher_trajectories" / task_name / f"num_context-{num_context}")
        model = variant.model.tensorproduct_koopman
        model.latent_dim, model.tensor_rank = lifting_dim, tensor_rank
        model.lambda_ae, model.lambda_lat, model.lambda_end = lambda_ae, lambda_lat, lambda_end
        if evaluation_samples is not None:
            variant.evaluation.num_posterior_samples = evaluation_samples
        variant.logging.output_root = str(output_root / "runs" / task_name / name)
        variant.logging.run_name = "trial"

        config_path = output_root / "generated_configs" / task_name / f"{name}.yaml"
        save_resolved_config(variant, str(config_path))
        trial.set_user_attr("config_path", str(config_path))
        try:
            artifacts = run_train_tensorproduct_koopman(str(config_path))
            summary_path = artifacts.run_paths.base_dir / "evaluation" / "tensorproduct_koopman_summary.json"
            with summary_path.open(encoding="utf-8") as handle:
                summary = json.load(handle)
            for key, value in summary.items():
                if key.startswith("mean_"):
                    trial.set_user_attr(key, value)
            run_summary_path = artifacts.run_paths.base_dir / "run_summary.json"
            with run_summary_path.open(encoding="utf-8") as handle:
                run_summary = json.load(handle)
            for key in (
                "teacher_data_time_seconds",
                "training_time_seconds",
                "training_plus_teacher_data_time_seconds",
                "evaluation_time_seconds",
                "total_run_time_seconds",
                "teacher_data_loaded_from_cache",
            ):
                if key in run_summary:
                    trial.set_user_attr(key, run_summary[key])
            trial.set_user_attr("run_dir", str(artifacts.run_paths.base_dir))
            trial.set_user_attr("checkpoint_path", str(artifacts.checkpoint_path))
            trial.set_user_attr("evaluation_summary_path", str(summary_path))
            trial.set_user_attr("per_observation_metrics_path", str(artifacts.run_paths.base_dir / "evaluation" / "tensorproduct_koopman_per_observation.csv"))
            return (
                float(summary["mean_c2st"]),
                float(summary["mean_sampling_time_per_generated_sample_ms"]),
            )
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            trial.set_user_attr("traceback", traceback.format_exc())
            raise

    return objective


def run_search(config_path: Path, requested_trials: int | None, dry_run: bool) -> Path:
    raw = load_yaml(config_path)
    relative_root = Path(str(raw.get("output_root", "logs/tensorproduct_optuna_search")))
    output_root = relative_root if relative_root.is_absolute() else ROOT / relative_root
    trial_limit = requested_trials if requested_trials is not None else int(raw.get("n_trials", 24))
    if trial_limit < 1:
        raise ValueError("n_trials must be positive.")
    tasks = raw.get("tasks")
    if not isinstance(tasks, list) or not tasks:
        raise ValueError("tasks must be a non-empty list.")
    positive_choices(raw["num_context"], "num_context")
    positive_choices(raw.get("lifting_dim", raw.get("latent_dim")), "lifting_dim")
    positive_choices(raw["tensor_rank"], "tensor_rank")
    print(f"Prepared {trial_limit} adaptive trials per task across {len(tasks)} task(s).")
    if dry_run:
        return output_root

    output_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_root / 'optuna_study.sqlite3'}"
    seed, startup = int(raw.get("seed", 0)), int(raw.get("sampler_startup_trials", 8))
    for index, task in enumerate(tasks):
        if not isinstance(task, dict) or "name" not in task or "config" not in task:
            raise ValueError("Each task needs name and config fields.")
        task_name = str(task["name"])
        candidate = Path(str(task["config"]))
        task_config = candidate if candidate.is_absolute() else (config_path.parent / candidate).resolve()
        study = optuna.create_study(
            study_name=f"tensorproduct_koopman_{task_name}",
            storage=storage,
            load_if_exists=True,
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=seed + index, n_startup_trials=startup),
        )
        remaining = max(0, trial_limit - len(study.trials))
        print(f"{task_name}: {len(study.trials)}/{trial_limit} recorded; running {remaining}.")
        if remaining:
            study.optimize(make_objective(task_name, task_config, raw, output_root), n_trials=remaining, catch=(Exception,))
        export_study(task_name, study, output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Tune tensor-product Koopman posterior recovery with Optuna.")
    parser.add_argument("--config", required=True, help="YAML Optuna search definition.")
    parser.add_argument("--trials", type=int, help="Total Optuna trials per task, including existing trials.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration without running trials.")
    args = parser.parse_args()
    print(f"Search results: {run_search(Path(args.config).resolve(), args.trials, args.dry_run)}")


if __name__ == "__main__":
    main()
