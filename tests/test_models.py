import torch

from koopman_sbi.config import CMPEModelConfig, FlowMatchingModelConfig, KoopmanModelConfig, NetworkConfig
from koopman_sbi.models import ConditionalFlowMatching, ConsistencyModelPosteriorEstimator, KoopmanFlow


def test_koopman_model_shapes():
    network = NetworkConfig(hidden_dims=[8, 8])
    config = KoopmanModelConfig(network=network, lifting_dim=6)
    model = KoopmanFlow(input_dim=2, context_dim=2, model_config=config, device=torch.device("cpu"))
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {"total_loss", "prediction_loss", "reconstruction_loss", "latent_loss"}
    assert torch.isfinite(losses["total_loss"])
    samples = model.sample_batch(torch.randn(4, 2))
    assert samples.shape == (4, 2)


def test_flow_matching_shapes():
    network = NetworkConfig(hidden_dims=[8, 8])
    config = FlowMatchingModelConfig(network=network, sigma_min=1e-4, time_prior_exponent=2.0)
    model = ConditionalFlowMatching(input_dim=2, context_dim=2, model_config=config, device=torch.device("cpu"))
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {"total_loss", "flow_matching_loss"}
    assert torch.isfinite(losses["total_loss"])
    samples = model.sample_batch(torch.randn(4, 2))
    assert samples.shape == (4, 2)


def test_cmpe_shapes_and_round_trip(tmp_path):
    network = NetworkConfig(hidden_dims=[8, 8])
    config = CMPEModelConfig(
        network=network,
        eps=1e-3,
        t_max=200.0,
        rho=7.0,
        sigma_data=1.0,
        s0=10,
        s1=50,
        p_mean=-1.1,
        p_std=2.0,
        default_num_steps=10,
    )
    model = ConsistencyModelPosteriorEstimator(
        input_dim=2,
        context_dim=2,
        model_config=config,
        device=torch.device("cpu"),
    )
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {"total_loss", "consistency_loss"}
    assert torch.isfinite(losses["total_loss"])
    samples_10 = model.sample_batch(torch.randn(4, 2), num_steps=10)
    samples_100 = model.sample_batch(torch.randn(4, 2), num_steps=100)
    assert samples_10.shape == (4, 2)
    assert samples_100.shape == (4, 2)

    checkpoint_path = tmp_path / "cmpe.pt"
    model.save(str(checkpoint_path))
    restored = ConsistencyModelPosteriorEstimator.load(str(checkpoint_path), device=torch.device("cpu"))
    restored_samples = restored.sample_batch(torch.randn(4, 2), num_steps=10)
    assert restored_samples.shape == (4, 2)
