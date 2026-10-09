import io
import json

import pytest
from PIL import Image

from app import (
    UploadValidationError,
    load_analysis_artifacts,
    low_confidence_message,
    validate_upload,
)


def _image_bytes(format_name: str) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), (20, 130, 40)).save(buffer, format=format_name)
    return buffer.getvalue()


@pytest.mark.parametrize(
    ("name", "format_name"), [("leaf.jpg", "JPEG"), ("leaf.png", "PNG")]
)
def test_valid_jpg_and_png_uploads_are_decoded(name, format_name):
    image = validate_upload(name, _image_bytes(format_name))

    assert image.size == (32, 24)
    assert image.mode == "RGB"


def test_text_payload_with_jpg_suffix_has_chinese_error():
    with pytest.raises(UploadValidationError, match="无法读取.*图片"):
        validate_upload("fake.jpg", b"plain text")


def test_unsupported_upload_suffix_is_rejected():
    with pytest.raises(UploadValidationError, match="仅支持"):
        validate_upload("leaf.gif", _image_bytes("PNG"))


def test_low_confidence_prediction_maps_to_required_warning():
    assert low_confidence_message(True) == "模型把握较低，图片可能不属于训练分布。"
    assert low_confidence_message(False) is None


def test_missing_analysis_files_remain_optional(tmp_path):
    artifacts = load_analysis_artifacts(tmp_path)

    assert artifacts.metrics is None
    assert artifacts.comparison is None
    assert artifacts.training_curve is None
    assert artifacts.confusion_matrix is None
    assert artifacts.errors is None


def test_available_analysis_file_loads_without_requiring_the_others(tmp_path):
    (tmp_path / "metrics.json").write_text(
        json.dumps({"model_name": "resnet18", "macro_f1": 0.8}), encoding="utf-8"
    )

    artifacts = load_analysis_artifacts(tmp_path)

    assert artifacts.metrics == {"model_name": "resnet18", "macro_f1": 0.8}
    assert artifacts.comparison is None
    assert artifacts.confusion_matrix is None


def test_analysis_artifacts_can_come_from_documented_separate_locations(tmp_path):
    evaluation_dir = tmp_path / "evaluation"
    training_dir = tmp_path / "training"
    comparison_path = tmp_path / "comparison.csv"
    evaluation_dir.mkdir()
    training_dir.mkdir()
    (evaluation_dir / "metrics.json").write_text(
        json.dumps({"model_name": "resnet18"}), encoding="utf-8"
    )
    (training_dir / "training_curves.png").write_bytes(b"png")
    comparison_path.write_text("model_name,macro_f1\nresnet18,0.8\n", encoding="utf-8")

    artifacts = load_analysis_artifacts(
        evaluation_dir,
        training_dir=training_dir,
        comparison_path=comparison_path,
    )

    assert artifacts.metrics == {"model_name": "resnet18"}
    assert artifacts.training_curve == training_dir / "training_curves.png"
    assert artifacts.comparison is not None
    assert artifacts.comparison.iloc[0]["model_name"] == "resnet18"
