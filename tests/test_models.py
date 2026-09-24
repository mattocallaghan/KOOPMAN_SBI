import torch

from koopman_sbi.config import (
    AdversarialConfig,
    CMPEModelConfig,
    FlowMatchingModelConfig,
    KoopmanModelConfig,
    NPEModelConfig,
    NetworkConfig,
    TensorProductKoopmanModelConfig,
)
from koopman_sbi.models import (
    ConditionalFlowMatching,
    ConsistencyModelPosteriorEstimator,
    KoopmanFlow,
    NormalizingFlowNPE,
    TensorProductKoopmanFlow,
)


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
    assert set(losses.keys()) == {
        "total_loss",
        "koopman_loss",
        "prediction_loss",
        "reconstruction_loss",
        "latent_loss",
    }
    assert torch.isfinite(losses["total_loss"])
    samples = model.sample_batch(torch.randn(4, 2))
    assert samples.shape == (4, 2)


def test_koopman_model_with_adversarial_loss():
    network = NetworkConfig(hidden_dims=[8, 8])
    config = KoopmanModelConfig(
        network=network,
        lifting_dim=6,
        adversarial=AdversarialConfig(enabled=True, lambda_adv=0.01, hidden_dims=[8, 8]),
    )
    model = KoopmanFlow(input_dim=2, context_dim=2, model_config=config, device=torch.device("cpu"))
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {
        "total_loss",
        "koopman_loss",
        "prediction_loss",
        "reconstruction_loss",
        "latent_loss",
        "generator_adversarial_loss",
        "discriminator_loss",
    }
    assert torch.isfinite(losses["total_loss"])


def test_tensorproduct_koopman_model_shapes_and_round_trip(tmp_path):
    state_network = NetworkConfig(hidden_dims=[8, 8])
    context_network = NetworkConfig(hidden_dims=[8])
    decoder_network = NetworkConfig(hidden_dims=[8, 8])
    config = TensorProductKoopmanModelConfig(
        state_network=state_network,
        context_network=context_network,
        decoder_network=decoder_network,
        latent_dim=6,
        context_feature_dim=5,
        tensor_rank=4,
    )
    model = TensorProductKoopmanFlow(
        input_dim=2,
        context_dim=3,
        model_config=config,
        device=torch.device("cpu"),
    )
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
        torch.randn(4, 3),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {
        "total_loss",
        "autoencoder_loss",
        "latent_loss",
        "endpoint_loss",
    }
    assert torch.isfinite(losses["total_loss"])
    samples = model.sample_batch(torch.randn(4, 3))
    assert samples.shape == (4, 2)

    checkpoint_path = tmp_path / "tensorproduct_koopman.pt"
    model.save(str(checkpoint_path))
    restored = TensorProductKoopmanFlow.load(str(checkpoint_path), device=torch.device("cpu"))
    assert restored.model_config.latent_dim == 6
    assert restored.model_config.context_feature_dim == 5
    assert restored.model_config.tensor_rank == 4
    assert restored.sample_batch(torch.randn(4, 3)).shape == (4, 2)


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


def test_npe_batched_sampling_matches_single_context_sampling():
    network = NetworkConfig(hidden_dims=[8, 8])
    config = NPEModelConfig(network=network, num_coupling_layers=2)
    model = NormalizingFlowNPE(input_dim=2, context_dim=2, model_config=config, device=torch.device("cpu"))
    context = torch.randn(5, 2)
    _ = model.flow.log_prob(torch.zeros_like(context), context=context)

    torch.manual_seed(123)
    expected = torch.cat(
        [
            model.flow.sample(num_samples=1, context=context[index : index + 1])[0].reshape(1, 2)
            for index in range(len(context))
        ],
        dim=0,
    )
    torch.manual_seed(123)
    actual = model.sample_batch(context)

    assert actual.shape == (5, 2)
    assert torch.allclose(actual, expected, atol=1e-6)


def test_nsf_shapes_and_round_trip(tmp_path):
    network = NetworkConfig(hidden_dims=[8, 8])
    config = NPEModelConfig(network=network, num_coupling_layers=2, transform="neural_spline")
    model = NormalizingFlowNPE(input_dim=2, context_dim=2, model_config=config, device=torch.device("cpu"))
    batch = (
        torch.randn(4, 2),
        torch.randn(4, 2),
    )
    losses = model.compute_loss(batch)
    assert set(losses.keys()) == {"total_loss", "negative_log_likelihood"}
    assert torch.isfinite(losses["total_loss"])
    samples = model.sample_batch(torch.randn(4, 2))
    assert samples.shape == (4, 2)

    checkpoint_path = tmp_path / "nsf.pt"
    model.save(str(checkpoint_path))
    restored = NormalizingFlowNPE.load(str(checkpoint_path), device=torch.device("cpu"))
    assert restored.model_config.transform == "neural_spline"
    assert restored.sample_batch(torch.randn(4, 2)).shape == (4, 2)


def test_cmpe_shapes_and_round_trip(tmp_path):
    network = NetworkConfig(hidden_dims=[8, 8], activation="mish", dropout=0.05)
    config = CMPEModelConfig(
        network=network,
        eps=1e-3,
        t_max=80.0,
        rho=7.0,
        sigma_data=1.0,
        s0=10,
        s1=150,
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
