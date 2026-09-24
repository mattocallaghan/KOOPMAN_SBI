from koopman_sbi.config import load_experiment_config


def test_load_config_defaults(tiny_config_path):
    config = load_experiment_config(str(tiny_config_path))
    assert config.task.name == "two_moons"
    assert config.model.koopman.lifting_dim == 12
    assert config.model.koopman.adversarial.enabled is False
    assert config.model.koopman.adversarial.lambda_adv == 0.01
    assert config.training.flow_matching.optimizer.lr == 1e-3
    assert config.evaluation.observations == [1, 2]
    assert config.model.cmpe.default_num_steps == 10
    assert config.training.cmpe.batch_size == 8
    assert config.benchmark_suite.variants[0].name == "npe"


def test_load_config_with_base_config(tmp_path):
    base_path = tmp_path / "base.yaml"
    base_path.write_text(
        """
task:
  name: two_moons
  seed: 0
  num_train_samples: 64
  simulation_batch_size: 16
  train_fraction: 0.75
model:
  flow_matching:
    network:
      hidden_dims: [8, 8]
  koopman:
    lifting_dim: 12
    lambda_rec: 1.0
    lambda_lat: 1.0
    lambda_pred: 1.0
    network:
      hidden_dims: [8, 8]
teacher: {}
training:
  flow_matching: {}
  koopman: {}
evaluation: {}
logging: {}
""",
        encoding="utf-8",
    )
    child_path = tmp_path / "child.yaml"
    child_path.write_text(
        """
base_config: "base.yaml"
model:
  koopman:
    lifting_dim: 16
""",
        encoding="utf-8",
    )

    config = load_experiment_config(str(child_path))
    assert config.task.name == "two_moons"
    assert config.model.koopman.lifting_dim == 16
    assert config.model.koopman.lambda_pred == 1.0
