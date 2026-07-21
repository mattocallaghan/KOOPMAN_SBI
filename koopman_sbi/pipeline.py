import argparse

from koopman_sbi.experiments import run_distill_koopman


def main() -> None:
    parser = argparse.ArgumentParser(description="Compatibility wrapper for distill-koopman.")
    parser.add_argument("--config", help="Path to experiment YAML config.")
    parser.add_argument("--koopman_config", help="Legacy config argument; now expects unified config.")
    parser.add_argument("--flow_config", help="Legacy config argument; now expects unified config.")
    parser.add_argument("--pipeline_config", help="Legacy config argument; now expects unified config.")
    args = parser.parse_args()

    config_path = args.config or args.pipeline_config or args.koopman_config or args.flow_config
    if config_path is None:
        parser.error("Provide --config or one of the legacy config flags.")
    run_distill_koopman(config_path)


if __name__ == "__main__":
    main()
