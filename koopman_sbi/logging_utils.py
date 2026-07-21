from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
from typing import Dict, Optional

from torch.utils.tensorboard import SummaryWriter

from koopman_sbi.config import ExperimentConfig


class ExperimentLogger:
    def __init__(
        self,
        config: ExperimentConfig,
        run_dir: Path,
        experiment_name: str,
    ) -> None:
        self.config = config
        self.run_dir = run_dir
        self.experiment_name = experiment_name
        self.tensorboard_writer: Optional[SummaryWriter] = None
        self.wandb_run = None

        if config.logging.use_tensorboard:
            tb_dir = run_dir / "tensorboard"
            tb_dir.mkdir(parents=True, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(str(tb_dir))

        if config.logging.use_wandb:
            try:
                import wandb

                self.wandb_run = wandb.init(
                    project=config.logging.wandb_project,
                    name=config.logging.wandb_run_name,
                    dir=str(run_dir),
                    config=asdict(config),
                    tags=config.logging.wandb_tags,
                    reinit=True,
                )
            except ImportError:
                print("wandb is not installed; continuing without wandb logging.")

    def log_metrics(self, metrics: Dict[str, float], step: int, prefix: str) -> None:
        if self.tensorboard_writer is not None:
            for key, value in metrics.items():
                self.tensorboard_writer.add_scalar(f"{prefix}/{key}", value, step)
        if self.wandb_run is not None:
            self.wandb_run.log({f"{prefix}/{key}": value for key, value in metrics.items()}, step=step)

    def log_image(self, name: str, image_path: Path) -> None:
        if self.wandb_run is not None:
            import wandb

            self.wandb_run.log({name: wandb.Image(str(image_path))})

    def log_run_summary(self, summary: Dict[str, float | int | str | bool]) -> None:
        summary_path = self.run_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(summary, handle, indent=2)
        if self.wandb_run is not None:
            self.wandb_run.summary.update(summary)

    def close(self) -> None:
        if self.tensorboard_writer is not None:
            self.tensorboard_writer.close()
        if self.wandb_run is not None:
            self.wandb_run.finish()
