from __future__ import annotations

import argparse

from koopman_sbi.config import resolve_task_config_path
from koopman_sbi.experiments import (
    run_gpu_evaluation,
    run_benchmark_compare,
    run_benchmark_suite,
    run_distill_koopman,
    run_evaluate,
    run_train_cmpe,
    run_train_flow,
    run_train_npe,
    run_train_koopman,
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Koopman SBI experiment CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    for command in [
        "train-flow",
        "train-npe",
        "train-cmpe",
        "train-koopman",
        "distill-koopman",
        "benchmark-compare",
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
            subparser.add_argument("--num-posterior-samples", type=int, default=1000)
            subparser.add_argument(
                "--observation-counts",
                type=int,
                nargs="+",
                default=[1, 10, 100, 1000, 10000, 100000, 1000000, 10000000],
            )
            subparser.add_argument("--max-context-batch-size", type=int, default=100000)
            subparser.add_argument(
                "--auto-max-context-batch-size",
                action="store_true",
                help="Auto-tune the largest feasible context batch size via OOM backoff before benchmarking.",
            )
            subparser.add_argument("--num-repeats", type=int, default=3)
            subparser.add_argument("--warmup-observations", type=int, default=1)
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
    elif args.command == "train-cmpe":
        run_train_cmpe(config_path)
    elif args.command == "train-koopman":
        run_train_koopman(config_path)
    elif args.command == "distill-koopman":
        run_distill_koopman(config_path)
    elif args.command == "benchmark-compare":
        run_benchmark_compare(config_path)
    elif args.command == "benchmark-suite":
        run_benchmark_suite(config_path)
    elif args.command == "evaluate":
        run_evaluate(config_path)
    elif args.command == "gpu-evaluation":
        run_gpu_evaluation(
            config_path=config_path,
            num_posterior_samples=args.num_posterior_samples,
            observation_counts=args.observation_counts,
            max_context_batch_size=args.max_context_batch_size,
            auto_max_context_batch_size=args.auto_max_context_batch_size,
            num_repeats=args.num_repeats,
            warmup_observations=args.warmup_observations,
        )
    else:
        parser.error(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
