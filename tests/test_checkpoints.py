from dataclasses import asdict

import pytest
import torch

from config import CLASS_IDS
from models.simple_cnn import SimpleCNN
from utils.checkpoints import (
    CheckpointCompatibilityError,
    CheckpointMetadata,
    load_checkpoint,
    save_checkpoint,
)


def _metadata(**overrides) -> CheckpointMetadata:
    values = {
        "schema_version": 1,
        "model_name": "simple_cnn",
        "class_ids": CLASS_IDS,
        "image_size": 224,
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        "training_config": {"batch_size": 32},
        "best_epoch": 4,
        "best_val_metrics": {"macro_f1": 0.81},
    }
    values.update(overrides)
    return CheckpointMetadata(**values)


def test_checkpoint_round_trip_preserves_state_and_metadata(tmp_path):
    model = SimpleCNN(num_classes=10)
    path = tmp_path / "nested" / "best.pth"
    metadata = _metadata()

    save_checkpoint(path, model, metadata)
    state_dict, loaded = load_checkpoint(
        path, expected_model="simple_cnn", expected_classes=CLASS_IDS
    )

    assert path.is_file()
    assert loaded == metadata
    assert set(state_dict) == set(model.state_dict())
    assert not path.with_suffix(path.suffix + ".tmp").exists()


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        ({"metadata": asdict(_metadata())}, "state_dict"),
        ({"state_dict": {}}, "metadata"),
        (
            {"state_dict": {}, "metadata": asdict(_metadata(schema_version=99))},
            "schema version",
        ),
    ],
)
def test_malformed_checkpoints_are_rejected(tmp_path, payload, message):
    path = tmp_path / "invalid.pth"
    torch.save(payload, path)

    with pytest.raises(CheckpointCompatibilityError, match=message):
        load_checkpoint(path)


def test_wrong_model_name_is_rejected(tmp_path):
    path = tmp_path / "best.pth"
    save_checkpoint(path, SimpleCNN(10), _metadata())

    with pytest.raises(CheckpointCompatibilityError, match="model"):
        load_checkpoint(path, expected_model="resnet18")


def test_reversed_class_order_is_rejected(tmp_path):
    path = tmp_path / "best.pth"
    save_checkpoint(path, SimpleCNN(10), _metadata())

    with pytest.raises(CheckpointCompatibilityError, match="class order"):
        load_checkpoint(path, expected_classes=tuple(reversed(CLASS_IDS)))
