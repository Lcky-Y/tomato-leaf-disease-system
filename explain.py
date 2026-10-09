"""Grad-CAM explanations for the ResNet18 classifier."""

from __future__ import annotations

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from predict import InferenceBundle


class UnsupportedExplanationError(ValueError):
    """Raised when a model has no supported explanation target."""


def generate_gradcam(
    image: Image.Image,
    bundle: InferenceBundle,
    class_index: int | None = None,
) -> Image.Image:
    if bundle.metadata.model_name != "resnet18":
        raise UnsupportedExplanationError("Grad-CAM 目前仅支持 ResNet18 模型。")

    model = bundle.model
    target_layer = model.layer4[-1].conv2
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    gradient_handle = None

    def capture_activation(_module, _inputs, output):
        nonlocal gradient_handle
        activations.append(output)
        gradient_handle = output.register_hook(lambda gradient: gradients.append(gradient))

    forward_handle = target_layer.register_forward_hook(capture_activation)
    try:
        model.eval()
        tensor = bundle.transform(image.convert("RGB")).unsqueeze(0).to(bundle.device)
        model.zero_grad(set_to_none=True)
        logits = model(tensor)
        target_index = int(logits.argmax(dim=1).item()) if class_index is None else class_index
        if target_index < 0 or target_index >= logits.size(1):
            raise ValueError(f"class_index must be between 0 and {logits.size(1) - 1}")
        logits[0, target_index].backward()
        if not activations or not gradients:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients")

        activation = activations[-1].detach()
        gradient = gradients[-1].detach()
        channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((channel_weights * activation).sum(dim=1, keepdim=True))
        cam = F.interpolate(
            cam,
            size=(image.height, image.width),
            mode="bilinear",
            align_corners=False,
        )[0, 0]
        minimum = cam.min()
        maximum = cam.max()
        if float(maximum - minimum) > 0:
            cam = (cam - minimum) / (maximum - minimum)
        else:
            cam = torch.zeros_like(cam)
        array = (cam.cpu().numpy() * 255).round().astype(np.uint8)
        return Image.fromarray(array, mode="L")
    finally:
        forward_handle.remove()
        if gradient_handle is not None:
            gradient_handle.remove()


def overlay_heatmap(
    image: Image.Image,
    heatmap: Image.Image,
    alpha: float = 0.45,
) -> Image.Image:
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1")
    rgb = image.convert("RGB")
    heat = np.asarray(heatmap.convert("L").resize(rgb.size), dtype=np.uint8)
    zeros = np.zeros_like(heat)
    colored = Image.fromarray(np.stack((heat, zeros, 255 - heat), axis=2), mode="RGB")
    return Image.blend(rgb, colored, alpha)
