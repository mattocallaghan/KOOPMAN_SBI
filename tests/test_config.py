from koopman_sbi.config import load_experiment_config


def test_load_config_defaults(tiny_config_path):
    config = load_experiment_config(str(tiny_config_path))
    assert config.task.name == "two_moons"
    assert config.model.koopman.lifting_dim == 12
    assert config.training.flow_matching.optimizer.lr == 1e-3
    assert config.evaluation.observations == [1, 2]
    assert config.model.cmpe.default_num_steps == 10
    assert config.training.cmpe.batch_size == 8
    assert config.benchmark_suite.variants[0].name == "npe"
