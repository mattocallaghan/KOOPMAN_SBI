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
    backend: str = "native"
    eps: float = 1e-3
    t_max: float = 80.0
    rho: float = 7.0
    sigma_data: float = 1.0
    s0: int = 10
    s1: int = 150
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
    adversarial: "AdversarialConfig" = field(default_factory=lambda: AdversarialConfig())


@dataclass
class TensorProductKoopmanModelConfig:
    state_network: NetworkConfig
    context_network: NetworkConfig
    decoder_network: NetworkConfig
    latent_dim: int = 256
    context_feature_dim: int = 128
    tensor_rank: int = 64
    lambda_ae: float = 0.2
    lambda_lat: float = 0.608
    lambda_end: float = 1.0
    use_time_dependent_consistency: bool = False
    lambda_cons: float = 1.0
    # Continuous mode: cap on the spectral norm of the tensor-product generator (so on |eigenvalues|); <= 0 disables.
    continuous_final_time: float = 1.0
    # Discrete mode: endpoint loss e^T (I + w M) e = MSE + w * error in the teacher's noise coordinates, with
    # M = (J J^T + eps I)^-1 normalised to mean trace/d = 1, J = d theta / d noise of the teacher flow from the
    # variational equation (cached with the trajectories). Applies to the discrete endpoint loss and the
    # continuous target loss; skipped (with a notice) for full-trajectory batches.
    pullback_endpoint_metric: bool = True
    # w: weight of the noise-coordinate term relative to the MSE term.
    pullback_metric_weight: float = 1.0
    # eps = (value * median sigma_max(J))^2 caps the weight of very thin posterior directions; 0 = no cap.
    pullback_metric_epsilon: float = 0.0
    # How the flow Jacobian J = d theta / d noise enters the metric: "full" (exact J, O(d^2) per pair; low d),
    # "diagonal" (Hutchinson diag(J J^T), per-dimension posterior spread; O(d)), "trace" (Hutchinson log|det J|,
    # one isotropic weight per pair; O(1)), "finite_difference_trace" (the trace estimate with a forward
    # difference instead of a JVP; one forward pass over a doubled batch per step, much cheaper on MPS),
    # "vjp_sketch" (k adjoint probes J^-T r: an unbiased estimate of the full anisotropic pull-back; O(k d)).
    jacobian_type: str = "full"
    # Random probes for "diagonal" and "vjp_sketch" ("trace" variants use one probe per trajectory).
    hutchinson_probes: int = 4
    # Solver tolerance for "vjp_sketch" (its own pass; the trajectories keep the teacher's tolerance).
    vjp_sketch_tolerance: float = 1e-3
    # Pairs per chunk in the vjp_sketch pass (memory-bound: ~19 GB per 512 pairs for the camera U-Net on MPS,
    # ~6.5 GB on an A100). Not part of the cache key, so it can be changed without regenerating trajectories.
    vjp_sketch_batch_size: int = 512
    adversarial: "AdversarialConfig" = field(default_factory=lambda: AdversarialConfig())


@dataclass
class AdversarialConfig:
    enabled: bool = False
    lambda_adv: float = 0.01
    hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    activation: str = "gelu"
    batch_norm: bool = False
    dropout: float = 0.0


@dataclass
class NPEModelConfig:
    network: NetworkConfig
    backend: str = "normflows"
    num_coupling_layers: int = 6
    transform: str = "affine"
    permutation: Optional[str] = "swap"
    use_actnorm: bool = False
    base_distribution: str = "normal"
    spline_num_bins: int = 8
    spline_tail_bound: float = 3.0
    spline_init_identity: bool = True


def _default_npe_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[128, 128, 128])


def _default_tensorproduct_state_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[256, 256, 256])


def _default_tensorproduct_context_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[256, 256])


def _default_tensorproduct_decoder_network_config() -> NetworkConfig:
    return NetworkConfig(hidden_dims=[256, 256, 256])


def _default_tensorproduct_koopman_config() -> TensorProductKoopmanModelConfig:
    return TensorProductKoopmanModelConfig(
        state_network=_default_tensorproduct_state_network_config(),
        context_network=_default_tensorproduct_context_network_config(),
        decoder_network=_default_tensorproduct_decoder_network_config(),
    )


def _default_nsf_model_config() -> NPEModelConfig:
    return NPEModelConfig(
        network=_default_npe_network_config(),
        transform="neural_spline",
        spline_num_bins=8,
        spline_tail_bound=3.0,
        spline_init_identity=True,
    )


def _default_cmpe_network_config() -> NetworkConfig:
    return NetworkConfig(
        hidden_dims=[256, 256, 256, 256, 256],
        activation="mish",
        dropout=0.05,
    )


@dataclass
class ModelSection:
    flow_matching: FlowMatchingModelConfig
    koopman: KoopmanModelConfig
    tensorproduct_koopman: TensorProductKoopmanModelConfig = field(
        default_factory=_default_tensorproduct_koopman_config
    )
    npe: NPEModelConfig = field(default_factory=lambda: NPEModelConfig(network=_default_npe_network_config()))
    nsf: NPEModelConfig = field(default_factory=_default_nsf_model_config)
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
    gradient_clip_norm: Optional[float] = None
    precision: str = "float32"
    use_tensorboard: bool = True
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(default_factory=SchedulerConfig)


@dataclass
class TrainingSection:
    flow_matching: TrainingConfig
    koopman: TrainingConfig
    tensorproduct_koopman: TrainingConfig = field(default_factory=TrainingConfig)
    npe: TrainingConfig = field(default_factory=TrainingConfig)
    nsf: TrainingConfig = field(default_factory=TrainingConfig)
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
    generate_trajectories: bool = True
    num_samples: int = 100000
    num_context: int = 10000
    batch_size: int = 1000
    store_full_trajectory: bool = False
    trajectory_steps: int = 16
    cache_trajectories: bool = True
    load_cached_trajectories: bool = True
    trajectory_dir: Optional[str] = None


@dataclass
class EvaluationConfig:
    num_posterior_samples: int = 10000
    observations: List[int] = None
    metrics: List[str] = None
    flow_checkpoint_path: Optional[str] = None
    tensorproduct_koopman_checkpoint_path: Optional[str] = None
    koopman_checkpoint_path: Optional[str] = None
    npe_checkpoint_path: Optional[str] = None
    nsf_checkpoint_path: Optional[str] = None
    cmpe_checkpoint_path: Optional[str] = None
    include_npe: bool = False
    include_nsf: bool = False
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
        BenchmarkVariantConfig(name="nsf", model_type="nsf"),
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
        BenchmarkVariantConfig(name="tensorproduct_koopman", model_type="tensorproduct_koopman"),
    ]


@dataclass
class BenchmarkSuiteConfig:
    train_flow_matching: bool = True
    train_models: bool = True
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

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "default.yaml"


def _convert_value(field_type: Any, value: Any, key_path: str = "") -> Any:
    origin = get_origin(field_type)
    if is_dataclass(field_type):
        return _build_dataclass(field_type, value, key_path)
    if origin in (list, List):
        inner_type = get_args(field_type)[0]
        return [_convert_value(inner_type, item, f"{key_path}[{index}]") for index, item in enumerate(value)]
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


def _build_dataclass(cls: Type[T], data: Dict[str, Any], key_path: str = "") -> T:
    if not isinstance(data, dict):
        raise TypeError(f"Expected a mapping for {key_path or cls.__name__}, got {type(data).__name__}")
    if cls is TensorProductKoopmanModelConfig and "use_full_trajectory" in data:
        data = dict(data)
        legacy_value = data.pop("use_full_trajectory")
        if "use_time_dependent_consistency" not in data or not data["use_time_dependent_consistency"]:
            data["use_time_dependent_consistency"] = legacy_value

    prefix = f"{key_path}." if key_path else ""
    known_fields = {field.name for field in fields(cls)}
    unknown_keys = sorted(str(key) for key in data if key not in known_fields)
    if unknown_keys:
        raise ValueError(
            f"Unknown config key(s) {', '.join(prefix + key for key in unknown_keys)}; "
            f"valid keys for {cls.__name__}: {', '.join(sorted(known_fields))}"
        )

    values: Dict[str, Any] = {}
    type_hints = get_type_hints(cls)
    for field in fields(cls):
        field_type = type_hints.get(field.name, field.type)
        if field.name in data:
            values[field.name] = _convert_value(field_type, data[field.name], prefix + field.name)
        elif field.default is not MISSING:
            values[field.name] = field.default
        elif field.default_factory is not MISSING:  # type: ignore[attr-defined]
            values[field.name] = field.default_factory()  # type: ignore[misc]
        else:
            raise ValueError(f"Missing required config field: {prefix}{field.name}")
    return cls(**values)


def _deep_merge_config(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw_config(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        raw_config = yaml.safe_load(handle)
    if not isinstance(raw_config, dict):
        raise TypeError(f"Expected YAML mapping in {path}, got {type(raw_config).__name__}")
    base_config_path = raw_config.pop("base_config", None)
    if base_config_path is None:
        return raw_config
    resolved_base_path = Path(base_config_path)
    if not resolved_base_path.is_absolute():
        resolved_base_path = path.parent / resolved_base_path
    base_config = _load_raw_config(resolved_base_path.resolve())
    return _deep_merge_config(base_config, raw_config)


def load_experiment_config(path: str) -> ExperimentConfig:
    """Load a config layered as `configs/default.yaml` <- base_config chain <- `path`."""
    config_path = Path(path).resolve()
    raw_config = _load_raw_config(config_path)
    if config_path != DEFAULT_CONFIG_PATH:
        raw_config = _deep_merge_config(_load_raw_config(DEFAULT_CONFIG_PATH), raw_config)
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
