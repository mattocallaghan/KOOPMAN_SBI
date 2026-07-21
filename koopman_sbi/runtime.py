from __future__ import annotations

import random
from datetime import datetime
from typing import Any

import numpy as np
import torch


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def detect_device(requested_device: str) -> torch.device:
    if requested_device not in {"auto", "cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device setting: {requested_device}")
    if requested_device == "cuda":
        return torch.device("cuda")
    if requested_device == "mps":
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    if requested_device == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        torch.set_default_dtype(torch.float32)
        return torch.device("mps")
    return torch.device("cpu")


def move_batch_to_device(batch: Any, device: torch.device) -> Any:
    if isinstance(batch, torch.Tensor):
        return move_tensor_to_device(batch, device)
    if isinstance(batch, (tuple, list)):
        return type(batch)(move_batch_to_device(item, device) for item in batch)
    if isinstance(batch, dict):
        return {key: move_batch_to_device(value, device) for key, value in batch.items()}
    return batch


def move_tensor_to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    if device.type == "mps" and tensor.dtype == torch.float64:
        return tensor.to(device=device, dtype=torch.float32)
    return tensor.to(device=device)


def timestamped_run_name(prefix: str) -> str:
    return f"{prefix}_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"


def to_python_scalar(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.detach().cpu().item()
        return value.detach().cpu().tolist()
    if isinstance(value, np.generic):
        return value.item()
    return value
