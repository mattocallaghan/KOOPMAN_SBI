import torch

from koopman_sbi.data import Standardizer


def test_standardizer_round_trip():
    theta = torch.randn(16, 2)
    x = torch.randn(16, 2)
    standardizer = Standardizer.from_training_tensors(theta, x)
    theta_standardized = standardizer.standardize_theta(theta)
    x_standardized = standardizer.standardize_x(x)
    assert torch.allclose(standardizer.inverse_theta(theta_standardized), theta, atol=1e-5)
    assert torch.allclose(standardizer.inverse_x(x_standardized), x, atol=1e-5)
