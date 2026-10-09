import numpy as np
import pytest
import torch
from PIL import Image

from config import CLASS_IDS
from dataset import build_transform
from explain import UnsupportedExplanationError, generate_gradcam, overlay_heatmap
from models.resnet18 import build_resnet18
from models.simple_cnn import SimpleCNN
from predict import InferenceBundle
from utils.checkpoints import CheckpointMetadata


def _metadata(model_name):
    return CheckpointMetadata(
        schema_version=1,
        model_name=model_name,
        class_ids=CLASS_IDS,
        image_size=224,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        training_config={},
        best_epoch=1,
        best_val_metrics={"macro_f1": 0.5},
    )


def _bundle(model, model_name):
    metadata = _metadata(model_name)
    return InferenceBundle(
        model=model,
        metadata=metadata,
        transform=build_transform(False, 224),
        device=torch.device("cpu"),
    )


def test_gradcam_and_overlay_match_original_size_and_release_hooks():
    model = build_resnet18(num_classes=10, pretrained=False)
    bundle = _bundle(model, "resnet18")
    image = Image.new("RGB", (72, 48), (20, 140, 40))
    target_layer = model.layer4[-1].conv2
    hooks_before = len(target_layer._forward_hooks)

    heatmap = generate_gradcam(image, bundle)
    overlay = overlay_heatmap(image, heatmap)

    assert heatmap.size == image.size
    assert overlay.size == image.size
    assert np.isfinite(np.asarray(heatmap, dtype=float)).all()
    assert np.isfinite(np.asarray(overlay, dtype=float)).all()
    assert len(target_layer._forward_hooks) == hooks_before


def test_gradcam_releases_hooks_when_forward_raises(monkeypatch):
    model = build_resnet18(num_classes=10, pretrained=False)
    bundle = _bundle(model, "resnet18")
    target_layer = model.layer4[-1].conv2
    hooks_before = len(target_layer._forward_hooks)

    def fail_forward(_inputs):
        raise RuntimeError("forward failed")

    monkeypatch.setattr(model, "forward", fail_forward)

    with pytest.raises(RuntimeError, match="forward failed"):
        generate_gradcam(Image.new("RGB", (32, 32)), bundle)
    assert len(target_layer._forward_hooks) == hooks_before


def test_simple_cnn_explanation_is_rejected_with_chinese_message():
    bundle = _bundle(SimpleCNN(10), "simple_cnn")

    with pytest.raises(UnsupportedExplanationError, match="仅支持.*ResNet18"):
        generate_gradcam(Image.new("RGB", (32, 32)), bundle)
