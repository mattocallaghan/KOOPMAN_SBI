from __future__ import annotations

from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from koopman_sbi.config import ExperimentConfig, load_experiment_config, save_resolved_config
from koopman_sbi.data import DatasetBundle, load_or_generate_dataset
from koopman_sbi.evaluation import BenchmarkModelSpec, benchmark_models, evaluate_model, regenerate_benchmark_plots
from koopman_sbi.logging_utils import ExperimentLogger
from koopman_sbi.models import (
    BayesFlowConsistencyModel,
    ConditionalFlowMatching,
    ConsistencyModelPosteriorEstimator,
    KoopmanFlow,
    NormalizingFlowNPE,
    TensorProductKoopmanFlow,
)
from koopman_sbi.paths import RunPaths, prepare_run_directories, resolve_existing_run_dir
from koopman_sbi.runtime import detect_device, set_global_seed
from koopman_sbi.teacher import TeacherTrajectoryBundle, load_or_generate_teacher_trajectories
from koopman_sbi.training import Trainer


@dataclass
class ExperimentArtifacts:
    config: ExperimentConfig
    run_paths: RunPaths
    checkpoint_path: Path


def _prepare_bayesflow_environment() -> None:
    mpl_config_dir = Path.cwd() / ".cache" / "matplotlib"
    mpl_config_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_config_dir))
    os.environ["KERAS_BACKEND"] = "torch"


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


def _resolve_nsf_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if config.evaluation.nsf_checkpoint_path and Path(config.evaluation.nsf_checkpoint_path).exists():
        return Path(config.evaluation.nsf_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "train_nsf"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "train_nsf"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    raise FileNotFoundError("No NSF checkpoint is available.")


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


def _load_cmpe_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    backend = checkpoint.get("backend", "native")
    if backend == "bayesflow":
        return BayesFlowConsistencyModel.load(str(checkpoint_path), device=device)
    return ConsistencyModelPosteriorEstimator.load(str(checkpoint_path), device=device)


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
        teacher_checkpoint_path=teacher_checkpoint,
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


def run_train_tensorproduct_koopman(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.tensorproduct_koopman.device)
    dataset_bundle = load_or_generate_dataset(config)
    teacher_checkpoint = _resolve_teacher_checkpoint(config, config_path)
    teacher_model = ConditionalFlowMatching.load(str(teacher_checkpoint), device=device)

    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="train_tensorproduct_koopman",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "train_tensorproduct_koopman")

    teacher_bundle = load_or_generate_teacher_trajectories(
        config=config,
        dataset_bundle=dataset_bundle,
        teacher_model=teacher_model,
        device=device,
        teacher_checkpoint_path=teacher_checkpoint,
    )
    model = TensorProductKoopmanFlow(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.tensorproduct_koopman,
        device=device,
    )
    model.to(device)

    train_loader, val_loader = _make_teacher_loaders(teacher_bundle, config.training.tensorproduct_koopman)
    trainer = Trainer(model, config.training.tensorproduct_koopman, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = TensorProductKoopmanFlow.load(str(result.best_checkpoint_path), device=device)
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="tensorproduct_koopman",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "train_tensorproduct_koopman",
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


def run_train_nsf(config_path: str) -> ExperimentArtifacts:
    run_start_time = time.time()
    config = _load_config(config_path)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.nsf.device)
    dataset_bundle = load_or_generate_dataset(config)
    run_paths = prepare_run_directories(
        output_root=config.logging.output_root,
        task_name=config.task.name,
        experiment_name="train_nsf",
        run_name=config.logging.run_name,
    )
    _save_run_config(config, run_paths)
    logger = _create_logger(config, run_paths, "train_nsf")

    model_config = config.model.nsf
    model_config.transform = "neural_spline"
    model = NormalizingFlowNPE(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=model_config,
        device=device,
    )
    model.to(device)
    train_loader, val_loader = _make_pair_loaders(config, dataset_bundle, config.training.nsf)
    trainer = Trainer(model, config.training.nsf, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = NormalizingFlowNPE.load(str(result.best_checkpoint_path), device=device)
    evaluation_start_time = time.time()
    evaluate_model(
        model=best_model,
        model_name="nsf",
        config=config,
        dataset_bundle=dataset_bundle,
        output_dir=run_paths.base_dir / "evaluation",
        logger=logger,
    )
    evaluation_time_seconds = time.time() - evaluation_start_time
    logger.log_run_summary(
        {
            "experiment_name": "train_nsf",
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
    if config.model.cmpe.backend == "bayesflow":
        return _run_train_cmpe_bayesflow(config, dataset_bundle, device, run_paths, logger, run_start_time)
    model = ConsistencyModelPosteriorEstimator(
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.cmpe,
        device=device,
    )
    model.to(device)
    train_loader, val_loader = _make_pair_loaders(config, dataset_bundle, config.training.cmpe)
    model.set_total_training_steps(config.training.cmpe.epochs * max(len(train_loader), 1))
    trainer = Trainer(model, config.training.cmpe, run_paths, logger)
    result = trainer.fit(train_loader, val_loader)

    best_model = ConsistencyModelPosteriorEstimator.load(str(result.best_checkpoint_path), device=device)
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


def _run_train_cmpe_bayesflow(
    config: ExperimentConfig,
    dataset_bundle: DatasetBundle,
    device: torch.device,
    run_paths: RunPaths,
    logger: ExperimentLogger,
    run_start_time: float,
) -> ExperimentArtifacts:
    setup_start = time.perf_counter()
    _prepare_bayesflow_environment()
    print("[cmpe/bayesflow] Importing BayesFlow and Keras...")
    import bayesflow as bf
    import keras
    print(f"[cmpe/bayesflow] Import complete in {time.perf_counter() - setup_start:.2f}s")

    train_dataset = dataset_bundle.train_dataset
    val_dataset = dataset_bundle.val_dataset
    batch_size = int(config.training.cmpe.batch_size)
    epochs = int(config.training.cmpe.epochs)
    num_training_batches = max((len(train_dataset) + batch_size - 1) // batch_size, 1)
    total_steps = num_training_batches * epochs

    model_setup_start = time.perf_counter()
    print("[cmpe/bayesflow] Building ConsistencyModel and BasicWorkflow...")
    consistency_model = bf.networks.ConsistencyModel(
        total_steps=total_steps,
        subnet_kwargs={
            "dropout": float(config.model.cmpe.network.dropout),
            "widths": tuple(int(width) for width in config.model.cmpe.network.hidden_dims),
            "activation": config.model.cmpe.network.activation,
        },
        max_time=float(config.model.cmpe.t_max),
        sigma2=float(config.model.cmpe.sigma_data),
        eps=float(config.model.cmpe.eps),
        s0=int(config.model.cmpe.s0),
        s1=int(config.model.cmpe.s1),
    )
    adapter = (
        bf.adapters.Adapter()
        .to_array()
        .convert_dtype("float64", "float32")
        .rename("theta", "inference_variables")
        .rename("x", "inference_conditions")
    )
    workflow = bf.BasicWorkflow(
        simulator=None,
        adapter=adapter,
        inference_network=consistency_model,
        initial_learning_rate=float(config.training.cmpe.optimizer.lr),
        optimizer=keras.optimizers.AdamW,
        checkpoint_filepath=str(run_paths.checkpoints_dir),
        checkpoint_name="best_model",
        save_best_only=True,
    )
    print(f"[cmpe/bayesflow] Workflow setup complete in {time.perf_counter() - model_setup_start:.2f}s")
    data_prep_start = time.perf_counter()
    print("[cmpe/bayesflow] Materializing offline training arrays...")
    training_data = {
        "theta": train_dataset.theta_raw.numpy().astype("float32"),
        "x": train_dataset.x_raw.numpy().astype("float32"),
    }
    validation_data = {
        "theta": val_dataset.theta_raw.numpy().astype("float32"),
        "x": val_dataset.x_raw.numpy().astype("float32"),
    }
    print(f"[cmpe/bayesflow] Offline arrays ready in {time.perf_counter() - data_prep_start:.2f}s")
    build_batch_size = min(batch_size, len(train_dataset))
    build_data = {
        "theta": training_data["theta"][:build_batch_size],
        "x": training_data["x"][:build_batch_size],
    }
    build_start = time.perf_counter()
    print("[cmpe/bayesflow] Prebuilding approximator on a concrete batch...")
    workflow.approximator.build_from_data(workflow.approximator.adapter(build_data, batched=True))
    print(f"[cmpe/bayesflow] Prebuild complete in {time.perf_counter() - build_start:.2f}s")
    compile_start = time.perf_counter()
    print("[cmpe/bayesflow] Compiling approximator...")
    workflow.approximator.compile(
        optimizer=keras.optimizers.AdamW(
            learning_rate=float(config.training.cmpe.optimizer.lr),
            weight_decay=float(config.training.cmpe.optimizer.weight_decay),
        ),
        run_eagerly=True,
    )
    print(f"[cmpe/bayesflow] Compile complete in {time.perf_counter() - compile_start:.2f}s")
    fit_start = time.time()
    print("[cmpe/bayesflow] Entering fit_offline...")
    history = workflow.fit_offline(
        data=training_data,
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        verbose=int(config.training.cmpe.fit_verbose),
    )
    training_time_seconds = time.time() - fit_start

    approximator = workflow.approximator
    model = BayesFlowConsistencyModel(
        approximator=approximator,
        input_dim=dataset_bundle.dim_theta,
        context_dim=dataset_bundle.dim_x,
        model_config=config.model.cmpe,
        device=device,
        theta_mean=dataset_bundle.standardizer.theta_mean,
        theta_std=dataset_bundle.standardizer.theta_std,
        x_mean=dataset_bundle.standardizer.x_mean,
        x_std=dataset_bundle.standardizer.x_std,
    )
    best_checkpoint_path = run_paths.checkpoints_dir / "best_model.pt"
    model.save(str(best_checkpoint_path))

    from koopman_sbi.training import mirror_checkpoint_to_last_model, write_history_records

    history_records = _convert_bayesflow_history(history)
    write_history_records(run_paths, history_records)
    mirror_checkpoint_to_last_model(run_paths, best_checkpoint_path)
    keras_path = best_checkpoint_path.with_suffix(".keras")
    last_keras_path = run_paths.last_model_dir / keras_path.name
    last_keras_path.write_bytes(keras_path.read_bytes())
    weights_path = best_checkpoint_path.with_suffix(".weights.h5")
    if weights_path.exists():
        last_weights_path = run_paths.last_model_dir / weights_path.name
        last_weights_path.write_bytes(weights_path.read_bytes())

    best_model = BayesFlowConsistencyModel.load(str(best_checkpoint_path), device=device)
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
            "cmpe_backend": "bayesflow",
            "training_time_seconds": training_time_seconds,
            "evaluation_time_seconds": evaluation_time_seconds,
            "total_run_time_seconds": time.time() - run_start_time,
            "best_checkpoint_path": str(best_checkpoint_path),
        }
    )
    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=best_checkpoint_path)


def _convert_bayesflow_history(history) -> list[Dict[str, float]]:
    history_dict = getattr(history, "history", {})
    if not history_dict:
        return []
    epochs = max(len(values) for values in history_dict.values())
    records: list[Dict[str, float]] = []
    for epoch_index in range(epochs):
        record: Dict[str, float] = {"epoch": float(epoch_index + 1)}
        for key, values in history_dict.items():
            if epoch_index < len(values):
                record[key] = float(values[epoch_index])
        records.append(record)
    return records


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


def _resolve_tensorproduct_koopman_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    if (
        config.evaluation.tensorproduct_koopman_checkpoint_path
        and Path(config.evaluation.tensorproduct_koopman_checkpoint_path).exists()
    ):
        return Path(config.evaluation.tensorproduct_koopman_checkpoint_path)
    last_model_candidate = (
        Path(config.logging.output_root)
        / config.task.name
        / "last_model"
        / "train_tensorproduct_koopman"
        / "best_model.pt"
    )
    if last_model_candidate.exists():
        return last_model_candidate
    if config.logging.run_name:
        candidate = (
            Path(config.logging.output_root)
            / config.task.name
            / "train_tensorproduct_koopman"
            / config.logging.run_name
            / "checkpoints"
            / "best_model.pt"
        )
        if candidate.exists():
            return candidate
    return run_train_tensorproduct_koopman(config_path).checkpoint_path


def _resolve_or_train_npe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    try:
        return _resolve_npe_checkpoint(config, config_path)
    except FileNotFoundError:
        return run_train_npe(config_path).checkpoint_path


def _resolve_or_train_nsf_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    try:
        return _resolve_nsf_checkpoint(config, config_path)
    except FileNotFoundError:
        return run_train_nsf(config_path).checkpoint_path


def _resolve_or_train_cmpe_checkpoint(config: ExperimentConfig, config_path: str) -> Path:
    try:
        return _resolve_cmpe_checkpoint(config, config_path)
    except FileNotFoundError:
        return run_train_cmpe(config_path).checkpoint_path


def _benchmark_suite_uses_model_type(config: ExperimentConfig, model_type: str) -> bool:
    return any(variant.model_type == model_type for variant in config.benchmark_suite.variants)


def _build_benchmark_suite_specs(
    config: ExperimentConfig,
    device: torch.device,
    flow_checkpoint: Path,
    koopman_checkpoint: Path,
    tensorproduct_koopman_checkpoint: Path | None = None,
    npe_checkpoint: Path | None = None,
    nsf_checkpoint: Path | None = None,
    cmpe_checkpoint: Path | None = None,
) -> list[BenchmarkModelSpec]:
    flow_model = ConditionalFlowMatching.load(str(flow_checkpoint), device=device)
    koopman_model = KoopmanFlow.load(str(koopman_checkpoint), device=device)
    tensorproduct_koopman_model = (
        TensorProductKoopmanFlow.load(str(tensorproduct_koopman_checkpoint), device=device)
        if tensorproduct_koopman_checkpoint is not None
        else None
    )
    npe_model = NormalizingFlowNPE.load(str(npe_checkpoint), device=device) if npe_checkpoint is not None else None
    nsf_model = NormalizingFlowNPE.load(str(nsf_checkpoint), device=device) if nsf_checkpoint is not None else None
    cmpe_model = (
        _load_cmpe_model(cmpe_checkpoint, device=device) if cmpe_checkpoint is not None else None
    )

    timing_lookup = {
        "flow_matching": _read_run_summary_from_checkpoint(flow_checkpoint),
        "koopman": _read_run_summary_from_checkpoint(koopman_checkpoint),
        "tensorproduct_koopman": (
            _read_run_summary_from_checkpoint(tensorproduct_koopman_checkpoint)
            if tensorproduct_koopman_checkpoint is not None
            else {}
        ),
        "npe": _read_run_summary_from_checkpoint(npe_checkpoint) if npe_checkpoint is not None else {},
        "nsf": _read_run_summary_from_checkpoint(nsf_checkpoint) if nsf_checkpoint is not None else {},
        "cmpe": _read_run_summary_from_checkpoint(cmpe_checkpoint) if cmpe_checkpoint is not None else {},
    }
    model_lookup = {
        "flow_matching": flow_model,
        "koopman": koopman_model,
        "tensorproduct_koopman": tensorproduct_koopman_model,
        "npe": npe_model,
        "nsf": nsf_model,
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


def run_benchmark_suite(
    config_path: str,
    *,
    plots_only: bool = False,
    force_retrain: bool = False,
) -> ExperimentArtifacts:
    config = _load_config(config_path)
    if plots_only:
        run_dir = resolve_existing_run_dir(
            output_root=config.logging.output_root,
            task_name=config.task.name,
            experiment_name="benchmark_suite",
            run_name=config.logging.run_name,
        )
        run_paths = RunPaths(
            base_dir=run_dir,
            checkpoints_dir=run_dir / "checkpoints",
            plots_dir=run_dir / "plots",
            metrics_dir=run_dir / "metrics",
            last_model_dir=Path(config.logging.output_root) / config.task.name / "last_model" / "benchmark_suite",
        )
        logger = _create_logger(config, run_paths, "benchmark_suite")
        regenerate_benchmark_plots(run_dir / "benchmark", config=config, logger=logger)
        logger.close()
        flow_checkpoint = _resolve_teacher_checkpoint(config, config_path)
        return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=flow_checkpoint)
    set_global_seed(config.task.seed)
    device = detect_device(config.training.flow_matching.device)
    dataset_bundle = load_or_generate_dataset(config)
    if force_retrain:
        flow_checkpoint = run_train_flow(config_path).checkpoint_path
        koopman_checkpoint = run_distill_koopman(config_path).checkpoint_path
        tensorproduct_koopman_checkpoint = (
            run_train_tensorproduct_koopman(config_path).checkpoint_path
            if _benchmark_suite_uses_model_type(config, "tensorproduct_koopman")
            else None
        )
        npe_checkpoint = run_train_npe(config_path).checkpoint_path
        nsf_checkpoint = run_train_nsf(config_path).checkpoint_path
        cmpe_checkpoint = run_train_cmpe(config_path).checkpoint_path
    else:
        flow_checkpoint = _resolve_teacher_checkpoint(config, config_path)
        koopman_checkpoint = _resolve_koopman_checkpoint(config, config_path)
        tensorproduct_koopman_checkpoint = (
            _resolve_tensorproduct_koopman_checkpoint(config, config_path)
            if _benchmark_suite_uses_model_type(config, "tensorproduct_koopman")
            else None
        )
        npe_checkpoint = _resolve_or_train_npe_checkpoint(config, config_path)
        nsf_checkpoint = _resolve_or_train_nsf_checkpoint(config, config_path)
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
        tensorproduct_koopman_checkpoint=tensorproduct_koopman_checkpoint,
        npe_checkpoint=npe_checkpoint,
        nsf_checkpoint=nsf_checkpoint,
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
    elif config.evaluation.tensorproduct_koopman_checkpoint_path:
        model = TensorProductKoopmanFlow.load(config.evaluation.tensorproduct_koopman_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.tensorproduct_koopman_checkpoint_path)
        evaluate_model(
            model,
            "tensorproduct_koopman",
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
    elif config.evaluation.nsf_checkpoint_path:
        model = NormalizingFlowNPE.load(config.evaluation.nsf_checkpoint_path, device=device)
        checkpoint_path = Path(config.evaluation.nsf_checkpoint_path)
        evaluate_model(
            model,
            "nsf",
            config,
            dataset_bundle,
            run_paths.base_dir / "evaluation",
            logger=logger,
        )
    elif config.evaluation.cmpe_checkpoint_path:
        model = _load_cmpe_model(Path(config.evaluation.cmpe_checkpoint_path), device=device)
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
            "evaluation.koopman_checkpoint_path, "
            "evaluation.tensorproduct_koopman_checkpoint_path, "
            "evaluation.npe_checkpoint_path, "
            "evaluation.nsf_checkpoint_path, or evaluation.cmpe_checkpoint_path."
        )

    logger.close()
    return ExperimentArtifacts(config=config, run_paths=run_paths, checkpoint_path=checkpoint_path)
