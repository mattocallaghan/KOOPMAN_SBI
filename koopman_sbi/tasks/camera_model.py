"""Noisy blurred camera task (the "camera model" experiment of Ramesh et al., GATSBI, ICLR 2022).

theta is a clean 28x28 EMNIST ("bymerge", train + test) character image with pixels in [0, 1]; the prior is
implicit, a uniform draw from the dataset. The simulator applies Poisson shot noise and then a Gaussian
point-spread blur (sigma = 3 pixels). The posterior has no closed form and no reference samples: evaluation
compares the student with the teacher flow, and posterior point estimates with the true image.

Parameters and data are flattened to 784-vectors so the task plugs into the same pipeline as SBIBM tasks;
`image_shape` lets image networks reshape them. The 12 held-out observations are the test pairs of the
original experiment (images kept in torchvision's EMNIST orientation, as there).
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F

IMAGE_SIDE = 28
PSF_WIDTH = 3.0
DEFAULT_EMNIST_ROOT = Path("logs/shared/emnist")
_OBSERVATIONS_PATH = Path(__file__).parent / "data" / "camera_model_observations.npz"
# The 12 held-out test pairs of the original experiment, used to rebuild _OBSERVATIONS_PATH when it is missing.
_OBSERVATIONS_URL = (
    "https://raw.githubusercontent.com/mackelab/gatsbi/main/plotting_code/plotting_data/camera_samples.npz"
)


def _download_observations(path: Path) -> None:
    """Fetch the original experiment's test pairs and save the true images and observations (12 x 784 each)."""
    import io
    import urllib.request

    print(f"[camera_model] {path} not found; downloading the test observations from {_OBSERVATIONS_URL}", flush=True)
    with urllib.request.urlopen(_OBSERVATIONS_URL, timeout=120) as response:
        archive = np.load(io.BytesIO(response.read()))
    theta = archive["theta_test"].astype(np.float32).reshape(-1, IMAGE_SIDE * IMAGE_SIDE)
    x = archive["obs_test"].astype(np.float32).reshape(-1, IMAGE_SIDE * IMAGE_SIDE)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, theta=theta, x=x)


def poisson_noise(images: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Poisson shot noise with the quantisation rule of skimage.util.random_noise(mode="poisson").

    Each image is scaled by vals = 2 ** ceil(log2(number of distinct pixel values)), Poisson-sampled,
    rescaled, and clipped to [0, 1]. images: (n, 784) in [0, 1].
    """
    sorted_values = images.sort(dim=1).values
    distinct = 1 + (sorted_values[:, 1:] != sorted_values[:, :-1]).sum(dim=1)
    vals = torch.pow(2.0, torch.ceil(torch.log2(distinct.to(images.dtype))))[:, None]
    noisy = torch.poisson(images * vals, generator=generator) / vals
    return noisy.clamp(0.0, 1.0)


def _gaussian_kernel(sigma: float, radius: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    offsets = torch.arange(-radius, radius + 1, dtype=dtype, device=device)
    kernel = torch.exp(-0.5 * (offsets / sigma) ** 2)
    return kernel / kernel.sum()


def _pad_symmetric(images: torch.Tensor, radius: int, dim: int) -> torch.Tensor:
    """scipy.ndimage mode="reflect": mirror including the edge pixel (d c b a | a b c d | d c b a)."""
    length = images.shape[dim]
    head = images.narrow(dim, 0, radius).flip(dim)
    tail = images.narrow(dim, length - radius, radius).flip(dim)
    return torch.cat([head, images, tail], dim=dim)


def gaussian_blur(images: torch.Tensor, sigma: float = PSF_WIDTH) -> torch.Tensor:
    """Separable Gaussian blur matching scipy.ndimage.gaussian_filter (truncate 4, mode "reflect").

    images: (n, 784).
    """
    radius = int(4.0 * sigma + 0.5)
    kernel = _gaussian_kernel(sigma, radius, images.dtype, images.device)
    grid = images.reshape(-1, 1, IMAGE_SIDE, IMAGE_SIDE)
    grid = F.conv2d(_pad_symmetric(grid, radius, dim=3), kernel.view(1, 1, 1, -1))
    grid = F.conv2d(_pad_symmetric(grid, radius, dim=2), kernel.view(1, 1, -1, 1))
    return grid.reshape(images.shape[0], -1)


def camera(images: torch.Tensor, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Noisy blurred grayscale camera: Poisson noise, then Gaussian point-spread blur."""
    return gaussian_blur(poisson_noise(images, generator=generator))


def load_emnist_images(root: Path = DEFAULT_EMNIST_ROOT, download: bool = True) -> torch.Tensor:
    """All EMNIST "bymerge" train + test images as (n, 784) float32 in [0, 1] (torchvision orientation)."""
    from torchvision.datasets import EMNIST

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    parts = [EMNIST(str(root), split="bymerge", train=train, download=download).data for train in (True, False)]
    return torch.cat(parts, dim=0).reshape(-1, IMAGE_SIDE * IMAGE_SIDE).to(torch.float32) / 255.0


class CameraModelTask:
    """SBIBM-like task object for the camera model (see module docstring)."""

    name = "camera_model"
    dim_parameters = IMAGE_SIDE * IMAGE_SIDE
    dim_data = IMAGE_SIDE * IMAGE_SIDE
    image_shape = (1, IMAGE_SIDE, IMAGE_SIDE)
    has_reference_posterior = False
    # One mean/std over all pixels (see data.Standardizer); per-pixel scaling blows up rare border pixels.
    standardization = "global"
    parameter_range = (0.0, 1.0)

    def __init__(self, emnist_root: Path = DEFAULT_EMNIST_ROOT) -> None:
        self.emnist_root = Path(emnist_root)
        self._images: Optional[torch.Tensor] = None
        if not _OBSERVATIONS_PATH.exists():
            _download_observations(_OBSERVATIONS_PATH)
        observations = np.load(_OBSERVATIONS_PATH)
        self._true_parameters = torch.from_numpy(observations["theta"])
        self._observations = torch.from_numpy(observations["x"])
        self.num_observations = len(self._observations)

    def _prior_images(self) -> torch.Tensor:
        """EMNIST images with the 12 held-out test images removed, so they are never simulated for training."""
        if self._images is None:
            images = load_emnist_images(self.emnist_root)
            is_test_image = torch.zeros(len(images), dtype=torch.bool)
            for start in range(0, len(images), 100_000):
                chunk = images[start:start + 100_000]
                is_test_image[start:start + len(chunk)] = torch.cdist(chunk, self._true_parameters).min(1).values < 0.02
            self._images = images[~is_test_image]
        return self._images

    def get_prior(self) -> Callable[[int], torch.Tensor]:
        def prior(num_samples: int) -> torch.Tensor:
            images = self._prior_images()
            return images[torch.randint(len(images), (int(num_samples),))]

        return prior

    def get_simulator(self) -> Callable[[torch.Tensor], torch.Tensor]:
        return lambda theta: camera(theta)

    def get_observation(self, num_observation: int) -> torch.Tensor:
        return self._observations[num_observation - 1].unsqueeze(0).clone()

    def get_true_parameters(self, num_observation: int) -> torch.Tensor:
        return self._true_parameters[num_observation - 1].unsqueeze(0).clone()
