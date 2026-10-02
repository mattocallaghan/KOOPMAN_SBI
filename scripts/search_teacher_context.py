#!/usr/bin/env python
"""Optuna search over teacher context coverage only."""

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

import optuna
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
cache_dir = ROOT / ".cache" / "matplotlib"
cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(cache_dir))
os.environ.setdefault("KERAS_BACKEND", "torch")

from koopman_sbi.config import load_experiment_config, save_resolved_config
from koopman_sbi.experiments import run_train_tensorproduct_koopman


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise TypeError("Search config must be a YAML mapping.")
    return value


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
    pareto = {trial.number for trial in study.best_trials}
    rows: list[dict[str, Any]] = []
    for trial in study.trials:
        values = trial.values or []
        rows.append(
            {
                "task": task_name,
                "trial_number": trial.number,
                "status": trial.state.name.lower(),
                "objective_mean_c2st": values[0] if len(values) > 0 else None,
                "objective_sampling_time_per_generated_sample_ms": values[1] if len(values) > 1 else None,
                "pareto_optimal": trial.number in pareto,
                **trial.params,
                **trial.user_attrs,
            }
        )
    rows.sort(key=lambda row: (row["status"] != "complete", row["objective_mean_c2st"] is None, row["objective_mean_c2st"] or float("inf")))
    write_rows(output_root / f"{task_name}_teacher_context_trials.csv", rows)


def make_objective(task_name: str, task_config: Path, raw: dict[str, Any], output_root: Path):
    context_choices = [int(value) for value in raw["num_context"]]
    evaluation_samples = raw.get("evaluation_num_posterior_samples")
    if evaluation_samples is not None:
        evaluation_samples = int(evaluation_samples)

    def objective(trial: optuna.Trial) -> tuple[float, float]:
        num_context = trial.suggest_categorical("num_context", context_choices)
        name = f"optuna-{trial.number:04d}-context-{num_context}"
        variant = copy.deepcopy(load_experiment_config(str(task_config)))
        variant.task.name = task_name
        variant.teacher.auto_train_if_missing = False
        variant.teacher.generate_trajectories = True
        variant.benchmark_suite.train_flow_matching = False
        variant.teacher.num_context = num_context
        variant.teacher.trajectory_dir = str(output_root / "teacher_trajectories" / task_name / f"num_context-{num_context}")
        if evaluation_samples is not None:
            variant.evaluation.num_posterior_samples = evaluation_samples
        variant.logging.output_root = str(output_root / "runs" / task_name / name)
        variant.logging.run_name = "trial"
        config_path = output_root / "generated_configs" / task_name / f"{name}.yaml"
        save_resolved_config(variant, str(config_path))
        trial.set_user_attr("config_path", str(config_path))
        try:
            artifacts = run_train_tensorproduct_koopman(str(config_path))
            evaluation_path = artifacts.run_paths.base_dir / "evaluation" / "tensorproduct_koopman_summary.json"
            run_summary_path = artifacts.run_paths.base_dir / "run_summary.json"
            with evaluation_path.open(encoding="utf-8") as handle:
                evaluation = json.load(handle)
            with run_summary_path.open(encoding="utf-8") as handle:
                run_summary = json.load(handle)
            for key, value in evaluation.items():
                if key.startswith("mean_"):
                    trial.set_user_attr(key, value)
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
            trial.set_user_attr("evaluation_summary_path", str(evaluation_path))
            trial.set_user_attr("per_observation_metrics_path", str(artifacts.run_paths.base_dir / "evaluation" / "tensorproduct_koopman_per_observation.csv"))
            return float(evaluation["mean_c2st"]), float(evaluation["mean_sampling_time_per_generated_sample_ms"])
        except Exception as error:
            trial.set_user_attr("error", f"{type(error).__name__}: {error}")
            trial.set_user_attr("traceback", traceback.format_exc())
            raise

    return objective


def run_search(config_path: Path, requested_trials: int | None, dry_run: bool) -> Path:
    raw = load_yaml(config_path)
    root_value = Path(str(raw.get("output_root", "logs/teacher_context_optuna_search")))
    output_root = root_value if root_value.is_absolute() else ROOT / root_value
    tasks = raw["tasks"]
    trial_limit = requested_trials if requested_trials is not None else int(raw.get("n_trials", len(raw["num_context"])))
    print(f"Prepared {trial_limit} adaptive trials per task across {len(tasks)} task(s).")
    if dry_run:
        return output_root
    output_root.mkdir(parents=True, exist_ok=True)
    storage = f"sqlite:///{output_root / 'optuna_study.sqlite3'}"
    seed = int(raw.get("seed", 0))
    for index, task in enumerate(tasks):
        task_name = str(task["name"])
        task_value = Path(str(task["config"]))
        task_config = task_value if task_value.is_absolute() else (config_path.parent / task_value).resolve()
        study = optuna.create_study(
            study_name=f"teacher_context_{task_name}",
            storage=storage,
            load_if_exists=True,
            directions=["minimize", "minimize"],
            sampler=optuna.samplers.TPESampler(seed=seed + index, n_startup_trials=min(3, trial_limit)),
        )
        remaining = max(0, trial_limit - len(study.trials))
        print(f"{task_name}: {len(study.trials)}/{trial_limit} recorded; running {remaining}.")
        if remaining:
            study.optimize(make_objective(task_name, task_config, raw, output_root), n_trials=remaining, catch=(Exception,))
        export_study(task_name, study, output_root)
    return output_root


def main() -> None:
    parser = argparse.ArgumentParser(description="Optuna search over teacher num_context.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--trials", type=int, help="Total trials per task, including existing trials.")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    print(f"Search results: {run_search(Path(args.config).resolve(), args.trials, args.dry_run)}")


if __name__ == "__main__":
    main()
