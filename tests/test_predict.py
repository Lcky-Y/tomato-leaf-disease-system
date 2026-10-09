import pytest
import torch
from PIL import Image
from torch import nn

from config import CLASS_IDS
from dataset import build_transform
from predict import InferenceBundle, predict_image
from utils.checkpoints import CheckpointMetadata


class RecordingModel(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.input_shapes = []

    def forward(self, inputs):
        self.input_shapes.append(tuple(inputs.shape))
        return self.fixed_logits.unsqueeze(0).repeat(inputs.size(0), 1)


def _metadata():
    return CheckpointMetadata(
        schema_version=1,
        model_name="simple_cnn",
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


def _bundle(logits):
    metadata = _metadata()
    return InferenceBundle(
        model=RecordingModel(logits),
        metadata=metadata,
        transform=build_transform(
            False,
            metadata.image_size,
            mean=metadata.normalization["mean"],
            std=metadata.normalization["std"],
        ),
        device=torch.device("cpu"),
    )


@pytest.mark.parametrize(
    "image",
    [
        Image.new("RGB", (40, 30), "green"),
        Image.new("L", (40, 30), 120),
        Image.new("RGBA", (40, 30), (0, 128, 0, 80)),
    ],
)
def test_rgb_grayscale_and_rgba_inputs_reach_model_as_three_channels(image):
    bundle = _bundle([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])

    result = predict_image(image, bundle)

    assert bundle.model.input_shapes == [(1, 3, 224, 224)]
    assert len(result.items) == 3


def test_top_three_is_sorted_and_uses_checkpoint_class_order():
    bundle = _bundle([0.0, 9.0, 1.0, 8.0, 2.0, 7.0, 3.0, 6.0, 4.0, 5.0])

    result = predict_image(Image.new("RGB", (20, 20)), bundle)

    assert [item.class_id for item in result.items] == [
        CLASS_IDS[1],
        CLASS_IDS[3],
        CLASS_IDS[5],
    ]
    assert [item.probability for item in result.items] == sorted(
        [item.probability for item in result.items], reverse=True
    )
    assert result.elapsed_ms >= 0


@pytest.mark.parametrize("top_k", [0, 11])
def test_top_k_outside_checkpoint_class_count_is_rejected(top_k):
    with pytest.raises(ValueError, match="top_k"):
        predict_image(Image.new("RGB", (20, 20)), _bundle([0.0] * 10), top_k=top_k)


def test_low_confidence_flag_uses_configured_threshold():
    result = predict_image(
        Image.new("RGB", (20, 20)),
        _bundle([0.0] * 10),
        confidence_threshold=0.50,
    )

    assert result.items[0].probability == pytest.approx(0.10)
    assert result.low_confidence is True
