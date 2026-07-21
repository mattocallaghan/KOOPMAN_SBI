from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

import sbibm
import torch

from koopman_sbi.config import ExperimentConfig, load_experiment_config, resolve_task_config_path
from koopman_sbi.models import ConditionalFlowMatching, KoopmanFlow, NormalizingFlowNPE
from koopman_sbi.runtime import detect_device


def _resolve_config_argument(args: argparse.Namespace) -> str:
    if args.config is not None:
        return args.config
    return str(resolve_task_config_path(args.task, args.config_dir))


def _infer_dimensions(task_name: str) -> tuple[int, int]:
    task = sbibm.get_task(task_name)
    prior_sample = task.get_prior()(1).float()
    observation = task.get_observation(num_observation=1).float()
    theta_dim = int(prior_sample.shape[-1])
    x_dim = int(observation.shape[-1])
    return theta_dim, x_dim


def _count_torch_parameters(module: torch.nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in module.parameters())
    trainable = sum(parameter.numel() for parameter in module.parameters() if parameter.requires_grad)
    return int(total), int(trainable)


def _format_rows(rows: Iterable[tuple[str, int, int]]) -> str:
    rows = list(rows)
    model_width = max(len("model"), *(len(name) for name, _, _ in rows))
    total_width = max(len("total_params"), *(len(f"{total:,}") for _, total, _ in rows))
    trainable_width = max(len("trainable_params"), *(len(f"{trainable:,}") for _, _, trainable in rows))
    header = (
        f"{'model'.ljust(model_width)}  "
        f"{'total_params'.rjust(total_width)}  "
        f"{'trainable_params'.rjust(trainable_width)}"
    )
    divider = "-" * len(header)
    body = [
        f"{name.ljust(model_width)}  {f'{total:,}'.rjust(total_width)}  {f'{trainable:,}'.rjust(trainable_width)}"
        for name, total, trainable in rows
    ]
    return "\n".join([header, divider, *body])


def build_models_for_config(config: ExperimentConfig) -> list[tuple[str, int, int]]:
    theta_dim, x_dim = _infer_dimensions(config.task.name)
    flow_device = detect_device(config.training.flow_matching.device)
    koopman_device = detect_device(config.training.koopman.device)
    npe_device = detect_device(config.training.npe.device)

    flow_model = ConditionalFlowMatching(
        input_dim=theta_dim,
        context_dim=x_dim,
        model_config=config.model.flow_matching,
        device=flow_device,
    )
    koopman_model = KoopmanFlow(
        input_dim=theta_dim,
        context_dim=x_dim,
        model_config=config.model.koopman,
        device=koopman_device,
    )
    npe_model = NormalizingFlowNPE(
        input_dim=theta_dim,
        context_dim=x_dim,
        model_config=config.model.npe,
        device=npe_device,
    )

    return [
        ("flow_matching", *_count_torch_parameters(flow_model)),
        ("koopman", *_count_torch_parameters(koopman_model)),
        ("npe", *_count_torch_parameters(npe_model)),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Print parameter counts for the Koopman SBI benchmark models.")
    config_group = parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument("--config", help="Path to experiment YAML config.")
    config_group.add_argument("--task", help="Benchmark task name, resolved via the task config directory.")
    parser.add_argument(
        "--config-dir",
        default=None,
        help="Optional directory containing per-task YAML configs. Defaults to `koopman_sbi/configs/tasks`.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    config_path = _resolve_config_argument(args)
    config = load_experiment_config(config_path)

    rows = build_models_for_config(config)
    print(f"config: {Path(config_path)}")
    print(f"task: {config.task.name}")
    print(_format_rows(rows))


if __name__ == "__main__":
    main()
