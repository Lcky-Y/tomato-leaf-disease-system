"""Deterministic setup and runtime metadata helpers."""

from __future__ import annotations

import platform
import random

import numpy as np
import torch
import torchvision


def seed_everything(seed: int) -> None:
    """Seed supported RNGs and prefer deterministic kernels.

    Deterministic kernels can reduce training throughput, but make course
    experiments easier to reproduce and compare.
    """

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def runtime_metadata() -> dict[str, str]:
    cuda_available = torch.cuda.is_available()
    return {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchvision": torchvision.__version__,
        "cuda_available": str(cuda_available),
        "cuda_version": str(torch.version.cuda or "not available"),
        "device_name": torch.cuda.get_device_name(0) if cuda_available else "CPU",
    }
