"""Versioned, self-describing checkpoint serialization."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import Tensor, nn


SCHEMA_VERSION = 1


class CheckpointCompatibilityError(ValueError):
    """Raised when a checkpoint cannot be used safely by this project."""


@dataclass(frozen=True)
class CheckpointMetadata:
    schema_version: int
    model_name: str
    class_ids: tuple[str, ...]
    image_size: int
    normalization: dict[str, list[float]]
    training_config: dict[str, object]
    best_epoch: int
    best_val_metrics: dict[str, float]


def save_checkpoint(
    path: Path,
    model: nn.Module,
    metadata: CheckpointMetadata,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    payload = {"state_dict": model.state_dict(), "metadata": asdict(metadata)}
    try:
        torch.save(payload, temporary)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _parse_metadata(raw: object) -> CheckpointMetadata:
    if not isinstance(raw, dict):
        raise CheckpointCompatibilityError("checkpoint metadata must be a mapping")
    required = {
        "schema_version",
        "model_name",
        "class_ids",
        "image_size",
        "normalization",
        "training_config",
        "best_epoch",
        "best_val_metrics",
    }
    missing = sorted(required - set(raw))
    if missing:
        raise CheckpointCompatibilityError(
            f"checkpoint metadata is missing keys: {', '.join(missing)}"
        )
    if raw["schema_version"] != SCHEMA_VERSION:
        raise CheckpointCompatibilityError(
            f"unsupported checkpoint schema version: {raw['schema_version']}"
        )
    try:
        return CheckpointMetadata(
            schema_version=int(raw["schema_version"]),
            model_name=str(raw["model_name"]),
            class_ids=tuple(raw["class_ids"]),
            image_size=int(raw["image_size"]),
            normalization={
                key: [float(value) for value in values]
                for key, values in dict(raw["normalization"]).items()
            },
            training_config=dict(raw["training_config"]),
            best_epoch=int(raw["best_epoch"]),
            best_val_metrics={
                key: float(value)
                for key, value in dict(raw["best_val_metrics"]).items()
            },
        )
    except (TypeError, ValueError, KeyError) as exc:
        raise CheckpointCompatibilityError("invalid checkpoint metadata types") from exc


def load_checkpoint(
    path: Path,
    expected_model: str | None = None,
    expected_classes: tuple[str, ...] | None = None,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, Tensor], CheckpointMetadata]:
    payload = torch.load(Path(path), map_location=map_location, weights_only=True)
    if not isinstance(payload, dict) or "state_dict" not in payload:
        raise CheckpointCompatibilityError("checkpoint is missing state_dict")
    if "metadata" not in payload:
        raise CheckpointCompatibilityError("checkpoint is missing metadata")
    if not isinstance(payload["state_dict"], dict):
        raise CheckpointCompatibilityError("checkpoint state_dict must be a mapping")

    metadata = _parse_metadata(payload["metadata"])
    if expected_model is not None and metadata.model_name != expected_model:
        raise CheckpointCompatibilityError(
            f"checkpoint model mismatch: expected {expected_model}, got {metadata.model_name}"
        )
    if expected_classes is not None and metadata.class_ids != expected_classes:
        raise CheckpointCompatibilityError("checkpoint class order mismatch")
    return payload["state_dict"], metadata
