"""Task lookup: SBIBM tasks, plus local tasks SBIBM does not provide (e.g. the camera model)."""

from __future__ import annotations

from typing import Any

import sbibm

from koopman_sbi.tasks.camera_model import CameraModelTask

LOCAL_TASKS = {CameraModelTask.name: CameraModelTask}


def get_task(name: str) -> Any:
    """Return the task object for `name`: a local task if one is registered, else the SBIBM task."""
    if name in LOCAL_TASKS:
        return LOCAL_TASKS[name]()
    return sbibm.get_task(name)


def has_reference_posterior(task: Any) -> bool:
    """SBIBM tasks ship reference posterior samples; implicit-prior tasks such as the camera model do not."""
    return bool(getattr(task, "has_reference_posterior", True))
