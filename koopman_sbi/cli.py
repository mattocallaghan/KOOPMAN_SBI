from __future__ import annotations

import argparse
import os
from pathlib import Path

os.environ.setdefault("KERAS_BACKEND", "torch")
mpl_config_dir = Path.cwd() / ".cache" / "matplotlib"
mpl_config_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))

from koopman_sbi.config import resolve_task_config_path
from koopman_sbi.experiments import (
    run_gpu_evaluation,
    run_benchmark_suite,
    run_distill_koopman,
    run_evaluate,
    run_train_cmpe,
    run_train_flow,
    run_train_npe,
    run_train_nsf,
    run_train_tensorproduct_koopman,
    run_train_koopman,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koopman SBI experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "train-flow",
        "train-npe",
        "train-nsf",
        "train-cmpe",
        "train-koopman",
        "train-tensorproduct-koopman",
        "distill-koopman",
        "benchmark-suite",
        "evaluate",
        "gpu-evaluation",
    ]:
        subparser = subparsers.add_parser(command)
        config_group = subparser.add_mutually_exclusive_group(required=True)
        config_group.add_argument("--config", help="Path to experiment YAML config.")
        config_group.add_argument("--task", help="Benchmark task name, resolved via the task config directory.")
        subparser.add_argument(
            "--config-dir",
            default=None,
            help="Optional directory containing per-task YAML configs. Defaults to `koopman_sbi/configs/tasks`.",
        )
        if command == "gpu-evaluation":
            subparser.add_argument(
                "--observation-counts",
                type=int,
                nargs="+",
                default=[1, 10, 100, 1000, 10000, 500000, 800000, 1000000, 2000000, 5000000, 10000000, 40000000, 100000000],
            )
            subparser.add_argument("--max-context-batch-size", type=int, default=100000)
            subparser.add_argument(
                "--auto-max-context-batch-size",
                action="store_true",
                help="Auto-tune the largest feasible context batch size via OOM backoff before benchmarking.",
            )
            subparser.add_argument("--num-repeats", type=int, default=3)
            subparser.add_argument("--warmup-observations", type=int, default=1)
            subparser.add_argument("--x-scale", choices=["log", "linear"], default="log")
            subparser.add_argument("--y-scale", choices=["log", "linear"], default="log")
        if command == "benchmark-suite":
            subparser.add_argument(
                "--plots-only",
                action="store_true",
                help="Regenerate benchmark plots from saved benchmark-suite artifacts without recomputing models.",
            )
            subparser.add_argument(
                "--force-retrain",
                action="store_true",
                help="Retrain requested benchmark models from scratch for this task before benchmarking.",
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
    if args.command == "train-flow":
        run_train_flow(config_path)
    elif args.command == "train-npe":
        run_train_npe(config_path)
    elif args.command == "train-nsf":
        run_train_nsf(config_path)
    elif args.command == "train-cmpe":
        run_train_cmpe(config_path)
    elif args.command == "train-koopman":
        run_train_koopman(config_path)
    elif args.command == "train-tensorproduct-koopman":
        run_train_tensorproduct_koopman(config_path)
    elif args.command == "distill-koopman":
        run_distill_koopman(config_path)
    elif args.command == "benchmark-suite":
        run_benchmark_suite(
            config_path,
            plots_only=bool(getattr(args, "plots_only", False)),
            force_retrain=bool(getattr(args, "force_retrain", False)),
        )
    elif args.command == "evaluate":
        run_evaluate(config_path)
    elif args.command == "gpu-evaluation":
        run_gpu_evaluation(
            config_path=config_path,
            num_posterior_samples=1,
            observation_counts=args.observation_counts,
            max_context_batch_size=args.max_context_batch_size,
            auto_max_context_batch_size=args.auto_max_context_batch_size,
            num_repeats=args.num_repeats,
            warmup_observations=args.warmup_observations,
            x_scale=args.x_scale,
            y_scale=args.y_scale,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
