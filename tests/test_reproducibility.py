import random

import numpy as np
import torch

from utils.reproducibility import runtime_metadata, seed_everything


def test_seed_everything_reproduces_python_numpy_and_torch_values():
    seed_everything(42)
    first = (random.random(), np.random.rand(), torch.rand(3))

    seed_everything(42)
    second = (random.random(), np.random.rand(), torch.rand(3))

    assert first[0] == second[0]
    assert first[1] == second[1]
    assert torch.equal(first[2], second[2])


def test_runtime_metadata_reports_required_environment_fields():
    metadata = runtime_metadata()

    assert {
        "python",
        "torch",
        "torchvision",
        "cuda_available",
        "cuda_version",
        "device_name",
    } <= set(metadata)
    assert all(isinstance(value, str) and value for value in metadata.values())
