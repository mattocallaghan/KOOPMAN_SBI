from __future__ import annotations

import argparse

from koopman_sbi.config import resolve_task_config_path
from koopman_sbi.experiments import run_benchmark_suite


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train and benchmark the full SBI model suite for a single sbibm task."
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
        "--plots-only",
        action="store_true",
        help="Regenerate benchmark-suite plots from saved artifacts without recomputing benchmarks.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Retrain requested benchmark models from scratch for this task before benchmarking.",
    )
    parser.add_argument(
        "--train-flow-matching",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override benchmark_suite.train_flow_matching for this run.",
    )
    parser.add_argument(
        "--generate-teacher",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Override teacher.generate_trajectories for this run.",
    )
    return parser


def _resolve_config_argument(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config
    return str(resolve_task_config_path(args.task, args.config_dir))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = _resolve_config_argument(args)
    run_benchmark_suite(
        config_path,
        plots_only=bool(args.plots_only),
        force_retrain=bool(args.force_retrain),
        train_flow_matching=args.train_flow_matching,
        generate_teacher=args.generate_teacher,
    )


if __name__ == "__main__":
    main()
