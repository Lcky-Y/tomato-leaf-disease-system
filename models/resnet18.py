"""ResNet18 construction and staged fine-tuning policy."""

from __future__ import annotations

from typing import Literal

from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def build_resnet18(num_classes: int, pretrained: bool = True) -> nn.Module:
    weights = ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def set_resnet_train_stage(
    model: nn.Module,
    stage: Literal["head", "layer4"],
) -> list[nn.Parameter]:
    if stage not in {"head", "layer4"}:
        raise ValueError("stage must be 'head' or 'layer4'")

    for parameter in model.parameters():
        parameter.requires_grad = False
    if stage == "layer4":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
    for parameter in model.fc.parameters():
        parameter.requires_grad = True
    return [parameter for parameter in model.parameters() if parameter.requires_grad]
