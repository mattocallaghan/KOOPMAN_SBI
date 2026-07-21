from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from koopman_sbi.config import ExperimentConfig, load_experiment_config, save_resolved_config
from koopman_sbi.data import DatasetBundle, load_or_generate_dataset
from koopman_sbi.evaluation import BenchmarkModelSpec, benchmark_models, evaluate_model
from koopman_sbi.logging_utils import ExperimentLogger
from koopman_sbi.models import (
    ConditionalFlowMatching,
    ConsistencyModelPosteriorEstimator,
    KoopmanFlow,
    NormalizingFlowNPE,
)
from koopman_sbi.paths import RunPaths, prepare_run_directories
from koopman_sbi.runtime import detect_device, set_global_seed
from koopman_sbi.teacher import TeacherTrajectoryBundle, load_or_generate_teacher_trajectories
from koopman_sbi.training import Trainer


@dataclass
class ExperimentArtifacts:
    config: ExperimentConfig
    run_paths: RunPaths
    checkpoint_path: Path


def _load_config(config_path: str) -> ExperimentConfig:
    return load_experiment_config(config_path)


def _create_logger(config: ExperimentConfig, run_paths: RunPaths, experiment_name: str) -> ExperimentLogger:
    return ExperimentLogger(config=config, run_dir=run_paths.base_dir, experiment_name=experiment_name)


def _save_run_config(config: ExperimentConfig, run_paths: RunPaths) -> None:
    save_resolved_config(config, str(run_paths.base_dir / "resolved_config.yaml"))


def _read_run_summary_from_checkpoint(checkpoint_path: Path) -> Dict[str, float | int | str | bool]:
    run_summary_path = checkpoint_path.parent.parent / "run_summary.json"
    if run_summary_path.exists():
        with open(run_summary_path, "r", encoding="utf-8") as handle:
            return json.load(handle)
    latest_run_path = checkpoint_path.parent / "latest_run.txt"
    if latest_run_path.exists():
        referenced_run_dir = Path(latest_run_path.read_text(encoding="utf-8").strip())
        referenced_summary_path = referenced_run_dir / "run_summary.json"
        if referenced_summary_path.exists():
            with open(referenced_summary_path, "r", encoding="utf-8") as handle:
                return json.load(handle)
    return {}


def _make_pair_loaders(config, dataset_bundle: DatasetBundle, training_cfg):
    train_loader = DataLoader(
        dataset_bundle.train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
    )
    val_loader = DataLoader(
        dataset_bundle.val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
    )
    return train_loader, val_loader


def _make_teacher_loaders(teacher_bundle: TeacherTrajectoryBundle, training_cfg):
    train_loader = DataLoader(
        teacher_bundle.train_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=True,
        num_workers=training_cfg.num_workers,
    )
    val_loader = DataLoader(
        teacher_bundle.val_dataset,
        batch_size=training_cfg.batch_size,
        shuffle=False,
        num_workers=training_cfg.num_workers,
    )
    return train_loader, val_loader


def run_train_flow(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="train_flow",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "train_flow")

    model = ConditionalFlowMatching(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.flow_matching,
        device=device,
    )
    model.to(device)

    train_loader, val_loader = _make_pair_loaders(config, dataset_bundle, config.training.flow_matching)
    trainer = Trainer(model, config.training.flow_matching, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = ConditionalFlowMatching.load(str(result.best_checkpoint_path), device=device)
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="flow_matching",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "train_flow",
            "training_time_seconds": result.training_time_seconds,
            "evaluation_time_seconds": evaluation_time_seconds,
            "total_run_time_seconds": time.time() - run_start_time,
            "best_checkpoint_path": str(result.best_checkpoint_path),
        }
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=result.best_checkpoint_path)


def _resolve_npe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.evaluation.npe_checkpoint_path and Path(config.evaluation.npe_checkpoint_path).exists():
        return Path(config.evaluation.npe_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "train_npe"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "train_npe"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No NPE checkpoint is available.")


def _resolve_cmpe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.evaluation.cmpe_checkpoint_path and Path(config.evaluation.cmpe_checkpoint_path).exists():
        return Path(config.evaluation.cmpe_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "train_cmpe"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "train_cmpe"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No CMPE checkpoint is available.")


def _resolve_teacher_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.teacher.checkpoint_path and Path(config.teacher.checkpoint_path).exists():
        return Path(config.teacher.checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "train_flow"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "train_flow"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    if not config.teacher.auto_train_if_missing:
        raise FileNotFoundError("Teacher checkpoint is required but was not provided.")
    flow_artifacts = run_train_flow(config_path)
    return flow_artifacts.checkpoint_path


def run_distill_koopman(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.koopman.device)
    dataset_bundle = load_or_generate_dataset(config)
    teacher_checkpoint = _resolve_teacher_checkpoint(config, config_path)
    teacher_model = ConditionalFlowMatching.load(str(teacher_checkpoint), device=device)

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="distill_koopman",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "distill_koopman")

    teacher_bundle = load_or_generate_teacher_trajectories(
        config=config,
        dataset_bundle=dataset_bundle,
        teacher_model=teacher_model,
        device=device,
    )
    koopman_model = KoopmanFlow(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.koopman,
        device=device,
    )
    koopman_model.to(device)

    train_loader, val_loader = _make_teacher_loaders(teacher_bundle, config.training.koopman)
    trainer = Trainer(koopman_model, config.training.koopman, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = KoopmanFlow.load(str(result.best_checkpoint_path), device=device)
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="koopman",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "distill_koopman",
            "teacher_data_time_seconds": teacher_bundle.generation_time_seconds,
            "teacher_data_loaded_from_cache": teacher_bundle.loaded_from_cache,
            "teacher_num_samples": teacher_bundle.num_samples,
            "teacher_num_context": teacher_bundle.num_context,
            "training_time_seconds": result.training_time_seconds,
            "training_plus_teacher_data_time_seconds": (
                teacher_bundle.generation_time_seconds + result.training_time_seconds
            ),
            "evaluation_time_seconds": evaluation_time_seconds,
            "total_run_time_seconds": time.time() - run_start_time,
            "best_checkpoint_path": str(result.best_checkpoint_path),
        }
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=result.best_checkpoint_path)


def run_train_koopman(config_path: str) -> ExperimentArtifacts:
    return run_distill_koopman(config_path)


def run_train_npe(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.npe.device)
    dataset_bundle = load_or_generate_dataset(config)
    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="train_npe",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "train_npe")

    model = NormalizingFlowNPE(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.npe,
        device=device,
    )
    model.to(device)
    train_loader, val_loader = _make_pair_loaders(config, dataset_bundle, config.training.npe)
    trainer = Trainer(model, config.training.npe, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = NormalizingFlowNPE.load(str(result.best_checkpoint_path), device=device)
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="npe",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "train_npe",
            "training_time_seconds": result.training_time_seconds,
            "evaluation_time_seconds": evaluation_time_seconds,
            "total_run_time_seconds": time.time() - run_start_time,
            "best_checkpoint_path": str(result.best_checkpoint_path),
        }
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=result.best_checkpoint_path)


def run_train_cmpe(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.cmpe.device)
    dataset_bundle = load_or_generate_dataset(config)

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="train_cmpe",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "train_cmpe")
    model = ConsistencyModelPosteriorEstimator(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.cmpe,
        device=device,
    )
    model.to(device)
    model.set_sigma2_from_theta(dataset_bundle.train_dataset.theta_raw)
    train_loader, val_loader = _make_pair_loaders(config, dataset_bundle, config.training.cmpe)
    model.set_total_training_steps(config.training.cmpe.epochs * max(len(train_loader), 1))
    trainer = Trainer(model, config.training.cmpe, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = ConsistencyModelPosteriorEstimator.load(str(result.best_checkpoint_path), device=device)
    best_model.set_sigma2_from_theta(dataset_bundle.train_dataset.theta_raw)
    best_model.set_total_training_steps(config.training.cmpe.epochs * max(len(train_loader), 1))
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="cmpe",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        sample_kwargs={"num_steps": config.model.cmpe.default_num_steps},
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "train_cmpe",
            "training_time_seconds": result.training_time_seconds,
            "evaluation_time_seconds": evaluation_time_seconds,
            "total_run_time_seconds": time.time() - run_start_time,
            "best_checkpoint_path": str(result.best_checkpoint_path),
        }
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=result.best_checkpoint_path)


def _resolve_koopman_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.evaluation.koopman_checkpoint_path and Path(config.evaluation.koopman_checkpoint_path).exists():
        return Path(config.evaluation.koopman_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "distill_koopman"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "distill_koopman"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    return run_distill_koopman(config_path).checkpoint_path


def _resolve_or_train_npe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    try:
        return _resolve_npe_checkpoint(config, config_path)
    except FileNotFoundError:
        return run_train_npe(config_path).checkpoint_path


def _resolve_or_train_cmpe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    try:
        return _resolve_cmpe_checkpoint(config, config_path)
    except FileNotFoundError:
        return run_train_cmpe(config_path).checkpoint_path


def _build_benchmark_suite_specs(
    config: ExperimentConfig,
    device: torch.device,
    flow_checkpoint: Path,
    koopman_checkpoint: Path,
    npe_checkpoint: Path | None,
    cmpe_checkpoint: Path | None,
) -> list[BenchmarkModelSpec]:
    flow_model = ConditionalFlowMatching.load(str(flow_checkpoint), device=device)
    koopman_model = KoopmanFlow.load(str(koopman_checkpoint), device=device)
    npe_model = NormalizingFlowNPE.load(str(npe_checkpoint), device=device) if npe_checkpoint is not None else None
    cmpe_model = (
        ConsistencyModelPosteriorEstimator.load(str(cmpe_checkpoint), device=device) if cmpe_checkpoint is not None else None
    )

    timing_lookup = {
        "flow_matching": _read_run_summary_from_checkpoint(flow_checkpoint),
        "koopman": _read_run_summary_from_checkpoint(koopman_checkpoint),
        "npe": _read_run_summary_from_checkpoint(npe_checkpoint) if npe_checkpoint is not None else {},
        "cmpe": _read_run_summary_from_checkpoint(cmpe_checkpoint) if cmpe_checkpoint is not None else {},
    }
    model_lookup = {
        "flow_matching": flow_model,
        "koopman": koopman_model,
        "npe": npe_model,
        "cmpe": cmpe_model,
    }

    benchmark_specs: list[BenchmarkModelSpec] = []
    for variant in config.benchmark_suite.variants:
        model = model_lookup.get(variant.model_type)
        if model is None:
            continue
        benchmark_specs.append(
            BenchmarkModelSpec(
                label=variant.name,
                model=model,
                sample_kwargs=dict(variant.sample_kwargs),
                timing_metadata=dict(timing_lookup.get(variant.model_type, {})),
            )
        )
    return benchmark_specs


def run_benchmark_compare(config_path: str) -> ExperimentArtifacts:
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    flow_checkpoint = (
        Path(config.evaluation.flow_checkpoint_path)
        if config.evaluation.flow_checkpoint_path and Path(config.evaluation.flow_checkpoint_path).exists()
        else _resolve_teacher_checkpoint(config, config_path)
    )
    koopman_checkpoint = _resolve_koopman_checkpoint(config, config_path)

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="benchmark_compare",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "benchmark_compare")

    benchmark_specs = [
        BenchmarkModelSpec(
            label="flow_matching",
            model=ConditionalFlowMatching.load(str(flow_checkpoint), device=device),
            timing_metadata=_read_run_summary_from_checkpoint(flow_checkpoint),
        ),
        BenchmarkModelSpec(
            label="koopman",
            model=KoopmanFlow.load(str(koopman_checkpoint), device=device),
            timing_metadata=_read_run_summary_from_checkpoint(koopman_checkpoint),
        ),
    ]
    if config.evaluation.include_npe or config.evaluation.npe_checkpoint_path:
        try:
            npe_checkpoint = _resolve_npe_checkpoint(config, config_path)
        except FileNotFoundError:
            npe_checkpoint = None
        if npe_checkpoint is not None:
            benchmark_specs.append(
                BenchmarkModelSpec(
                    label="npe",
                    model=NormalizingFlowNPE.load(str(npe_checkpoint), device=device),
                    timing_metadata=_read_run_summary_from_checkpoint(npe_checkpoint),
                )
            )
    benchmark_models(
        models=benchmark_specs,
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "benchmark",
        timing_metadata={spec.label: spec.timing_metadata for spec in benchmark_specs},
        logger=logger,
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=koopman_checkpoint)


def run_benchmark_suite(config_path: str) -> ExperimentArtifacts:
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    flow_checkpoint = _resolve_teacher_checkpoint(config, config_path)
    koopman_checkpoint = _resolve_koopman_checkpoint(config, config_path)
    npe_checkpoint = _resolve_or_train_npe_checkpoint(config, config_path)
    cmpe_checkpoint = _resolve_or_train_cmpe_checkpoint(config, config_path)

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="benchmark_suite",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "benchmark_suite")
    benchmark_specs = _build_benchmark_suite_specs(
        config=config,
        device=device,
        flow_checkpoint=flow_checkpoint,
        koopman_checkpoint=koopman_checkpoint,
        npe_checkpoint=npe_checkpoint,
        cmpe_checkpoint=cmpe_checkpoint,
    )
    benchmark_models(
        models=benchmark_specs,
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "benchmark",
        timing_metadata={spec.label: spec.timing_metadata for spec in benchmark_specs},
        logger=logger,
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=flow_checkpoint)


def run_evaluate(config_path: str) -> ExperimentArtifacts:
    config = _load_config(config_path)
    if (
        config.evaluation.flow_checkpoint_path and config.evaluation.koopman_checkpoint_path
    ) or config.evaluation.include_npe:
        return run_benchmark_compare(config_path)

    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="evaluate",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "evaluate")

    if config.evaluation.flow_checkpoint_path:
        model = ConditionalFlowMatching.load(config.evaluation.flow_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.flow_checkpoint_path)
        evaluate_model(
            model,
            "flow_matching",
            config,
            dataset_bundle,
            run_paths.base_dir / "evaluation",
            logger=logger,
        )
    elif config.evaluation.koopman_checkpoint_path:
        model = KoopmanFlow.load(config.evaluation.koopman_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.koopman_checkpoint_path)
        evaluate_model(
            model,
            "koopman",
            config,
            dataset_bundle,
            run_paths.base_dir / "evaluation",
            logger=logger,
        )
    elif config.evaluation.npe_checkpoint_path:
        model = NormalizingFlowNPE.load(config.evaluation.npe_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.npe_checkpoint_path)
        evaluate_model(
            model,
            "npe",
            config,
            dataset_bundle,
            run_paths.base_dir / "evaluation",
            logger=logger,
        )
    elif config.evaluation.cmpe_checkpoint_path:
        model = ConsistencyModelPosteriorEstimator.load(config.evaluation.cmpe_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.cmpe_checkpoint_path)
        evaluate_model(
            model,
            "cmpe",
            config,
            dataset_bundle,
            run_paths.base_dir / "evaluation",
            sample_kwargs={"num_steps": config.model.cmpe.default_num_steps},
            logger=logger,
        )
    else:
        raise FileNotFoundError(
            "Provide at least one checkpoint path in evaluation.flow_checkpoint_path, "
            "evaluation.koopman_checkpoint_path, evaluation.npe_checkpoint_path, "
            "or evaluation.cmpe_checkpoint_path."
        )

    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=checkpoint_path)
