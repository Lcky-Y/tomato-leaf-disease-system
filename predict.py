"""Checkpoint-backed single-image Top-K inference."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import torch
from PIL import Image
from torch import nn

from config import CLASS_NAMES_CN
from dataset import build_transform
from models import build_model
from utils.checkpoints import CheckpointCompatibilityError, CheckpointMetadata, load_checkpoint


@dataclass(frozen=True)
class InferenceBundle:
    model: nn.Module
    metadata: CheckpointMetadata
    transform: Callable
    device: torch.device


@dataclass(frozen=True)
class PredictionItem:
    class_id: str
    display_name: str
    probability: float


@dataclass(frozen=True)
class PredictionResult:
    items: tuple[PredictionItem, ...]
    elapsed_ms: float
    low_confidence: bool


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def load_inference_bundle(
    checkpoint_path: Path,
    device: str = "auto",
) -> InferenceBundle:
    resolved_device = _resolve_device(device)
    state_dict, metadata = load_checkpoint(checkpoint_path, map_location=resolved_device)
    unknown = [class_id for class_id in metadata.class_ids if class_id not in CLASS_NAMES_CN]
    if unknown:
        raise CheckpointCompatibilityError(
            f"checkpoint contains unknown class identifiers: {', '.join(unknown)}"
        )
    try:
        mean = metadata.normalization["mean"]
        std = metadata.normalization["std"]
    except KeyError as exc:
        raise CheckpointCompatibilityError(
            "checkpoint normalization requires mean and std"
        ) from exc
    model = build_model(
        metadata.model_name,
        num_classes=len(metadata.class_ids),
        pretrained=False,
    ).to(resolved_device)
    model.load_state_dict(state_dict)
    model.eval()
    return InferenceBundle(
        model=model,
        metadata=metadata,
        transform=build_transform(
            False, metadata.image_size, mean=mean, std=std
        ),
        device=resolved_device,
    )


def predict_image(
    image: Image.Image,
    bundle: InferenceBundle,
    top_k: int = 3,
    confidence_threshold: float = 0.50,
) -> PredictionResult:
    class_count = len(bundle.metadata.class_ids)
    if top_k < 1 or top_k > class_count:
        raise ValueError(f"top_k must be between 1 and {class_count}")
    tensor = bundle.transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
    bundle.model.eval()
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    started = time.perf_counter()
    with torch.no_grad():
        logits = bundle.model(tensor)
        probabilities = torch.softmax(logits, dim=1)[0]
    if bundle.device.type == "cuda":
        torch.cuda.synchronize(bundle.device)
    elapsed_ms = (time.perf_counter() - started) * 1000
    values, indices = probabilities.topk(top_k)
    items = tuple(
        PredictionItem(
            class_id=bundle.metadata.class_ids[index],
            display_name=CLASS_NAMES_CN.get(
                bundle.metadata.class_ids[index], bundle.metadata.class_ids[index]
            ),
            probability=float(value),
        )
        for value, index in zip(values.cpu().tolist(), indices.cpu().tolist())
    )
    return PredictionResult(
        items=items,
        elapsed_ms=elapsed_ms,
        low_confidence=items[0].probability < confidence_threshold,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    bundle = load_inference_bundle(args.checkpoint, args.device)
    with Image.open(args.image) as image:
        result = predict_image(image, bundle)
    for item in result.items:
        print(f"{item.display_name}: {item.probability:.2%}")
    if result.low_confidence:
        print("提示：模型把握较低，图片可能不属于训练分布。")


if __name__ == "__main__":
    main()
