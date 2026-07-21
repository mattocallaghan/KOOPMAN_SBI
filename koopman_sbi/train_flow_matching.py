import argparse

from koopman_sbi.experiments import run_train_flow


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for train-flow.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_train_flow(args.config)


if __name__ == "__main__":
    main()
