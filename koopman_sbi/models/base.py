from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict

import torch
import torch.nn as nn


class BasePosteriorModel(nn.Module, ABC):
    def __init__(self) -> None:
        super().__init__()
        self.optimizer = None
        self.scheduler = None

    @abstractmethod
    def compute_loss(self, batch: Any) -> Dict[str, torch.Tensor]:
        raise NotImplementedError

    @abstractmethod
    def sample_batch(self, context: torch.Tensor, **kwargs: Any) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def save(self, filepath: str) -> None:
        raise NotImplementedError
