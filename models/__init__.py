"""Model factory shared by training, evaluation, and inference."""

from __future__ import annotations

from torch import nn

from .resnet18 import build_resnet18
from .simple_cnn import SimpleCNN


VALID_MODEL_NAMES = ("simple_cnn", "resnet18")


def build_model(
    name: str,
    num_classes: int,
    pretrained: bool = True,
) -> nn.Module:
    if name == "simple_cnn":
        return SimpleCNN(num_classes=num_classes)
    if name == "resnet18":
        return build_resnet18(num_classes=num_classes, pretrained=pretrained)
    raise ValueError(
        f"Unknown model '{name}'. Valid choices: {', '.join(VALID_MODEL_NAMES)}"
    )
