import argparse

from koopman_sbi.experiments import run_train_koopman


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for train-koopman.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML config.")
    args = parser.parse_args()
    run_train_koopman(args.config)


if __name__ == "__main__":
    main()
