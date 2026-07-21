from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Type, TypeVar, Union, get_args, get_origin, get_type_hints

import yaml


@dataclass
class NetworkConfig:
    hidden_dims: List[int]
    activation: str = "gelu"
    batch_norm: bool = False
    dropout: float = 0.0
    theta_with_glu: bool = False
    context_with_glu: bool = False
    type: str = "DenseResidualNet"


@dataclass
class FlowMatchingModelConfig:
    network: NetworkConfig
    sigma_min: float = 1e-4
    time_prior_exponent: float = 4.0
    integration_steps: Optional[int] = None
    atol: float = 1e-5
    rtol: float = 1e-5


@dataclass
class CMPEModelConfig:
    network: NetworkConfig
    eps: float = 1e-3
    t_max: float = 200.0
    rho: float = 7.0
    sigma_data: float = 1.0
    s0: int = 10
    s1: int = 50
    p_mean: float = -1.1
    p_std: float = 2.0
    default_num_steps: int = 10


@dataclass
class KoopmanModelConfig:
    network: NetworkConfig
    lifting_dim: int = 256
    lambda_rec: float = 1.0
    lambda_lat: float = 1.0
    lambda_pred: float = 1.0


@dataclass
class NPEModelConfig:
    network: NetworkConfig
    backend: str = "normflows"
    num_coupling_layers: int = 6
    transform: str = "affine"
    permutation: Optional[str] = "swap"
    use_actnorm: bool = False
    base_distribution: str = "normal"


def _default_npe_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[128, 128, 128])


def _default_cmpe_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[256, 256, 256, 256, 256], activation="relu")


@dataclass
class ModelSection:
    flow_matching: FlowMatchingModelConfig
    koopman: KoopmanModelConfig
    npe: NPEModelConfig = field(default_factory=lambda: NPEModelConfig(network=_default_npe_network_config()))
    cmpe: CMPEModelConfig = field(default_factory=lambda: CMPEModelConfig(network=_default_cmpe_network_config()))


@dataclass
class OptimizerConfig:
    name: str = "Adam"
    lr: float = 1e-3
    weight_decay: float = 1e-5


@dataclass
class SchedulerConfig:
    type: str = "reduce_on_plateau"
    factor: float = 0.5
    patience: int = 5
    step_size: int = 10
    gamma: float = 0.5


@dataclass
class TrainingConfig:
    batch_size: int = 64
    epochs: int = 100
    fit_verbose: int = 2
    num_workers: int = 0
    device: str = "auto"
    early_stopping: bool = True
    patience: int = 20
    early_stopping_min_delta: float = 0.0
    gradient_clip_norm: Optional[float] = None
    precision: str = "float32"
    use_tensorboard: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class TrainingSection:
    flow_matching: TrainingConfig
    koopman: TrainingConfig
    npe: TrainingConfig = field(default_factory=TrainingConfig)
    cmpe: TrainingConfig = field(default_factory=TrainingConfig)


@dataclass
class TaskConfig:
    name: str = "two_moons"
    seed: int = 0
    num_train_samples: int = 100000
    simulation_batch_size: int = 256
    train_fraction: float = 0.8
    use_cached_dataset: bool = True
    dataset_dir: Optional[str] = None


@dataclass
class TeacherConfig:
    checkpoint_path: Optional[str] = None
    auto_train_if_missing: bool = True
    num_samples: int = 100000
    num_context: int = 10000
    batch_size: int = 1000
    cache_trajectories: bool = True
    load_cached_trajectories: bool = True
    trajectory_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    num_posterior_samples: int = 10000
    observations: List[int] = None
    metrics: List[str] = None
    flow_checkpoint_path: Optional[str] = None
    koopman_checkpoint_path: Optional[str] = None
    npe_checkpoint_path: Optional[str] = None
    cmpe_checkpoint_path: Optional[str] = None
    include_npe: bool = False
    save_observation_plots: bool = True

    def __post_init__(self) -> None:
        if self.observations is None:
            self.observations = list(range(1, 11))
        if self.metrics is None:
            self.metrics = ["c2st", "mmd", "posterior_mean_error", "posterior_variance_ratio"]


@dataclass
class LoggingConfig:
    output_root: str = "logs"
    run_name: Optional[str] = None
    use_tensorboard: bool = True
    use_wandb: bool = False
    wandb_project: str = "koopman-sbi"
    wandb_run_name: Optional[str] = None
    wandb_tags: List[str] = None

    def __post_init__(self) -> None:
        if self.wandb_tags is None:
            self.wandb_tags = []


@dataclass
class BenchmarkVariantConfig:
    name: str
    model_type: str
    sample_kwargs: Dict[str, Any] = field(default_factory=dict)


def _default_benchmark_variants() -> List["BenchmarkVariantConfig"]:
    return [
        BenchmarkVariantConfig(name="npe", model_type="npe"),
        BenchmarkVariantConfig(
            name="fmnpe_dopri5",
            model_type="flow_matching",
            sample_kwargs={"solver": "dopri5", "atol": 1e-5, "rtol": 1e-5},
        ),
        BenchmarkVariantConfig(
            name="fmnpe_rk4_10",
            model_type="flow_matching",
            sample_kwargs={"solver": "rk4", "integration_steps": 10},
        ),
        BenchmarkVariantConfig(
            name="fmnpe_rk4_100",
            model_type="flow_matching",
            sample_kwargs={"solver": "rk4", "integration_steps": 100},
        ),
        BenchmarkVariantConfig(
            name="fmnpe_rk4_1000",
            model_type="flow_matching",
            sample_kwargs={"solver": "rk4", "integration_steps": 1000},
        ),
        BenchmarkVariantConfig(name="cmpe_10", model_type="cmpe", sample_kwargs={"num_steps": 10}),
        BenchmarkVariantConfig(name="cmpe_100", model_type="cmpe", sample_kwargs={"num_steps": 100}),
        BenchmarkVariantConfig(name="koopman", model_type="koopman"),
    ]


@dataclass
class BenchmarkSuiteConfig:
    variants: List[BenchmarkVariantConfig] = field(default_factory=_default_benchmark_variants)


@dataclass
class ExperimentConfig:
    task: TaskConfig
    model: ModelSection
    teacher: TeacherConfig
    training: TrainingSection
    evaluation: EvaluationConfig
    logging: LoggingConfig
    benchmark_suite: BenchmarkSuiteConfig = field(default_factory=BenchmarkSuiteConfig)


T = TypeVar("T")


def _convert_value(field_type: Any, value: Any) -> Any:
    origin = get_origin(field_type)
    if is_dataclass(field_type):
        return _build_dataclass(field_type, value)
    if origin in (list, List):
        inner_type = get_args(field_type)[0]
        return [_convert_value(inner_type, item) for item in value]
    if origin in (dict, Dict):
        key_type, value_type = get_args(field_type)
        return {
            _convert_value(key_type, key): _convert_value(value_type, item)
            for key, item in value.items()
        }
    if origin is Union:
        union_args = [arg for arg in get_args(field_type) if arg is not type(None)]
        if value is None:
            return None
        for arg in union_args:
            try:
                return _convert_value(arg, value)
            except Exception:
                continue
    return value


def _build_dataclass(cls: Type[T], data: Dict[str, Any]) -> T:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {cls.__name__}, got {type(data).__name__}")

    values: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        field_type = type_hints.get(field.name, field.type)
        if field.name in data:
            values[field.name] = _convert_value(field_type, data[field.name])
        elif field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            values[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required config field: {cls.__name__}.{field.name}")
    return cls(**values)


def load_experiment_config(path: str) -> ExperimentConfig:
    with open(path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    return _build_dataclass(ExperimentConfig, raw_config)


def resolve_task_config_path(
    task_name: str,
    config_dir: str | None = None,
) -> Path:
    normalized_task_name = task_name.replace("-", "_")
    base_dir = Path(config_dir) if config_dir is not None else Path(__file__).resolve().parent / "configs" / "tasks"
    candidate_paths = [
        base_dir / f"{normalized_task_name}.yaml",
        base_dir / f"{task_name}.yaml",
    ]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return candidate_path
    searched = ", ".join(str(path) for path in candidate_paths)
    raise FileNotFoundError(f"No config found for task '{task_name}'. Looked in: {searched}")


def config_to_dict(config: ExperimentConfig) -> Dict[str, Any]:
    return asdict(config)


def save_resolved_config(config: ExperimentConfig, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        yaml.safe_dump(config_to_dict(config), handle, sort_keys=False)
