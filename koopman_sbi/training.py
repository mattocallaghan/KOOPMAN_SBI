from __future__ import annotations

import csv
import json
import math
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from koopman_sbi.config import OptimizerConfig, SchedulerConfig, TrainingConfig
from koopman_sbi.logging_utils import ExperimentLogger
from koopman_sbi.paths import RunPaths
from koopman_sbi.runtime import move_batch_to_device, to_python_scalar


@dataclass
class TrainingResult:
    best_checkpoint_path: Path
    history: List[Dict[str, float]]
    training_time_seconds: float


def write_history_records(run_paths: RunPaths, history: List[Dict[str, float]]) -> None:
    json_path = run_paths.metrics_dir / "training_history.json"
    csv_path = run_paths.metrics_dir / "training_history.csv"
    with open(json_path, "w", encoding="utf-8") as handle:
        json.dump(history, handle, indent=2)
    if history:
        with open(csv_path, "w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(history[0].keys()))
            writer.writeheader()
            writer.writerows(history)


def mirror_checkpoint_to_last_model(run_paths: RunPaths, best_checkpoint_path: Path) -> None:
    last_model_path = run_paths.last_model_dir / best_checkpoint_path.name
    shutil.copy2(best_checkpoint_path, last_model_path)
    sidecar_path = Path(str(best_checkpoint_path) + ".meta.json")
    if sidecar_path.exists():
        shutil.copy2(sidecar_path, Path(str(last_model_path) + ".meta.json"))
    latest_run_path = run_paths.last_model_dir / "latest_run.txt"
    with open(latest_run_path, "w", encoding="utf-8") as handle:
        handle.write(str(run_paths.base_dir) + "\n")


def build_optimizer(parameters, optimizer_config: OptimizerConfig):
    optimizer_class = getattr(optim, optimizer_config.name)
    return optimizer_class(
        parameters,
        lr=float(optimizer_config.lr),
        weight_decay=float(optimizer_config.weight_decay),
    )


def build_scheduler(optimizer, scheduler_config: SchedulerConfig):
    if scheduler_config.type == "reduce_on_plateau":
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_config.factor),
            patience=int(scheduler_config.patience),
        )
    if scheduler_config.type == "StepLR":
        return optim.lr_scheduler.StepLR(
            optimizer,
            step_size=int(scheduler_config.step_size),
            gamma=float(scheduler_config.gamma),
        )
    raise ValueError(f"Unsupported scheduler type: {scheduler_config.type}")


class Trainer:
    def __init__(
        self,
        model,
        training_config: TrainingConfig,
        run_paths: RunPaths,
        logger: ExperimentLogger,
    ) -> None:
        self.model = model
        self.training_config = training_config
        self.run_paths = run_paths
        self.logger = logger
        self.discriminator_optimizer = None
        self.discriminator_scheduler = None
        discriminator = getattr(self.model, "discriminator", None)
        generator_parameters = (
            model.generator_parameters()
            if hasattr(model, "generator_parameters")
            else model.parameters()
        )
        self.model.optimizer = build_optimizer(generator_parameters, training_config.optimizer)
        self.model.scheduler = build_scheduler(self.model.optimizer, training_config.scheduler)
        if discriminator is not None:
            discriminator_parameters = (
                model.discriminator_parameters()
                if hasattr(model, "discriminator_parameters")
                else discriminator.parameters()
            )
            self.discriminator_optimizer = build_optimizer(discriminator_parameters, training_config.optimizer)
            self.discriminator_scheduler = build_scheduler(
                self.discriminator_optimizer,
                training_config.scheduler,
            )

    def _run_epoch(self, dataloader: DataLoader, train: bool) -> Dict[str, float]:
        if train:
            self.model.train()
        else:
            self.model.eval()

        metrics_accumulator: Dict[str, float] = {}
        num_batches = 0
        for batch in dataloader:
            batch = move_batch_to_device(batch, self.model.device)
            with torch.set_grad_enabled(train):
                if train and hasattr(self.model, "train_batch"):
                    loss_dict = self.model.train_batch(
                        batch,
                        optimizer=self.model.optimizer,
                        discriminator_optimizer=self.discriminator_optimizer,
                        gradient_clip_norm=self.training_config.gradient_clip_norm,
                    )
                else:
                    loss_dict = self.model.compute_loss(batch, stage="training" if train else "validation")
                    if train:
                        self.model.optimizer.zero_grad()
                        loss_dict["total_loss"].backward()
                        if self.training_config.gradient_clip_norm is not None:
                            torch.nn.utils.clip_grad_norm_(
                                self.model.parameters(),
                                self.training_config.gradient_clip_norm,
                            )
                        self.model.optimizer.step()
            for key, value in loss_dict.items():
                metrics_accumulator[key] = metrics_accumulator.get(key, 0.0) + float(to_python_scalar(value))
            num_batches += 1

        return {key: value / max(num_batches, 1) for key, value in metrics_accumulator.items()}

    def fit(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingResult:
        fit_start_time = time.time()
        best_val_loss = float("inf")
        history: List[Dict[str, float]] = []
        best_checkpoint_path = self.run_paths.checkpoints_dir / "best_model.pt"

        for epoch in range(1, self.training_config.epochs + 1):
            start_time = time.time()
            train_metrics = self._run_epoch(train_loader, train=True)
            val_metrics = self._run_epoch(val_loader, train=False)
            epoch_time = time.time() - start_time

            nonfinite_metrics = {
                name: value
                for name, value in {
                    **{f"train_{key}": value for key, value in train_metrics.items()},
                    **{f"val_{key}": value for key, value in val_metrics.items()},
                }.items()
                if not math.isfinite(value)
            }
            if nonfinite_metrics:
                raise FloatingPointError(
                    "Training produced non-finite metrics at "
                    f"epoch {epoch}: {nonfinite_metrics}. "
                    "Adjust the model architecture or optimization settings before retrying."
                )

            if isinstance(self.model.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                self.model.scheduler.step(val_metrics["total_loss"])
            else:
                self.model.scheduler.step()
            if self.discriminator_scheduler is not None:
                if isinstance(self.discriminator_scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.discriminator_scheduler.step(val_metrics["total_loss"])
                else:
                    self.discriminator_scheduler.step()

            learning_rate = self.model.optimizer.param_groups[0]["lr"]
            record = {
                "epoch": epoch,
                "epoch_time_seconds": epoch_time,
                "learning_rate": learning_rate,
                **{f"train_{key}": value for key, value in train_metrics.items()},
                **{f"val_{key}": value for key, value in val_metrics.items()},
            }
            history.append(record)

            self.logger.log_metrics(train_metrics, epoch, "train")
            self.logger.log_metrics(val_metrics, epoch, "validation")
            self.logger.log_metrics(
                {"epoch_time_seconds": epoch_time, "learning_rate": learning_rate},
                epoch,
                "training",
            )

            val_loss = val_metrics["total_loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save(str(best_checkpoint_path))

            print(
                f"Epoch {epoch}/{self.training_config.epochs} "
                f"train_total_loss={train_metrics['total_loss']:.4g} "
                f"val_total_loss={val_loss:.4g} "
                f"best_val_loss={best_val_loss:.4g} "
                f"time={epoch_time:.2f}s"
            )

        self._write_history(history)
        self._update_last_model_checkpoint(best_checkpoint_path)
        return TrainingResult(
            best_checkpoint_path=best_checkpoint_path,
            history=history,
            training_time_seconds=time.time() - fit_start_time,
        )

    def _write_history(self, history: List[Dict[str, float]]) -> None:
        write_history_records(self.run_paths, history)

    def _update_last_model_checkpoint(self, best_checkpoint_path: Path) -> None:
        mirror_checkpoint_to_last_model(self.run_paths, best_checkpoint_path)
