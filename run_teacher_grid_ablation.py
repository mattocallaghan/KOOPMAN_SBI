from __future__ import annotations

import argparse

from koopman_sbi.ablations import run_teacher_grid_ablation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the two_moons Koopman teacher-grid ablation study.")
    parser.add_argument(
        "--config",
        default="koopman_sbi/configs/ablations/two_moons_teacher_grid_ablation.yaml",
        help="Path to the ablation YAML config.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    run_teacher_grid_ablation(args.config)


if __name__ == "__main__":
    main()
