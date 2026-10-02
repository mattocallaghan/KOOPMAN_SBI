import numpy as np
import pytest
import torch

from koopman_sbi.data import Standardizer
from koopman_sbi.models.image_networks import ConditionalConvUNet, ConvDecoder, ConvEncoder
from koopman_sbi.tasks.camera_model import CameraModelTask, gaussian_blur, poisson_noise


def test_blur_matches_scipy_gaussian_filter():
    ndimage = pytest.importorskip("scipy.ndimage")
    images = torch.rand(3, 784, dtype=torch.float64)
    reference = np.stack([ndimage.gaussian_filter(image.reshape(28, 28), sigma=3.0) for image in images.numpy()])
    assert np.allclose(gaussian_blur(images).numpy().reshape(3, 28, 28), reference, atol=1e-12)


def test_poisson_noise_is_unbiased_and_in_range():
    torch.manual_seed(0)
    image = torch.linspace(0.05, 0.8, 784).round(decimals=2)
    noisy = poisson_noise(image.repeat(4000, 1))
    assert noisy.min() >= 0.0 and noisy.max() <= 1.0
    assert torch.allclose(noisy.mean(0), image, atol=0.03)


def test_observations_are_consistent_with_true_parameters():
    task = CameraModelTask()
    theta, x = task.get_true_parameters(1), task.get_observation(1)
    assert theta.shape == x.shape == (1, 784)
    # x is a noisy blur of theta: much closer to blur(theta) than to theta itself.
    assert (x - gaussian_blur(theta)).abs().mean() < (x - theta).abs().mean()


def test_global_standardizer_round_trip():
    theta, x = torch.rand(32, 784), torch.rand(32, 784)
    standardizer = Standardizer.from_training_tensors(theta, x, per_dimension=False)
    assert torch.allclose(standardizer.theta_std, standardizer.theta_std[0].expand(784))
    assert torch.allclose(standardizer.inverse_theta(standardizer.standardize_theta(theta)), theta, atol=1e-5)


def test_image_network_shapes():
    unet = ConditionalConvUNet(784, 784, [8, 16])
    assert unet(torch.randn(4, 784 * 2 + 1)).shape == (4, 784)
    encoder = ConvEncoder(785, 32, [8, 16], num_pixels=784)
    assert encoder(torch.randn(4, 785)).shape == (4, 32)
    decoder = ConvDecoder(32, 784, [16, 8])
    assert decoder(torch.randn(4, 32)).shape == (4, 784)
