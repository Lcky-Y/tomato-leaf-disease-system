"""Streamlit entry point and pure UI support helpers."""

from __future__ import annotations

import io
import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import streamlit as st
from PIL import Image, UnidentifiedImageError

from predict import InferenceBundle, load_inference_bundle


ALLOWED_UPLOAD_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_UPLOAD_BYTES = 10 * 1024 * 1024
MAX_IMAGE_PIXELS = 20_000_000


class UploadValidationError(ValueError):
    """A Chinese-facing error for an invalid uploaded image."""


@dataclass(frozen=True)
class AnalysisArtifacts:
    metrics: dict[str, object] | None
    comparison: pd.DataFrame | None
    training_curve: Path | None
    confusion_matrix: Path | None
    errors: pd.DataFrame | None


def validate_upload(name: str, payload: bytes) -> Image.Image:
    suffix = Path(name).suffix.lower()
    if suffix not in ALLOWED_UPLOAD_SUFFIXES:
        raise UploadValidationError("仅支持 JPG、JPEG 或 PNG 图片。")
    if not payload:
        raise UploadValidationError("上传文件为空。")
    if len(payload) > MAX_UPLOAD_BYTES:
        raise UploadValidationError("图片超过 10 MB 上传限制。")
    try:
        with Image.open(io.BytesIO(payload)) as candidate:
            candidate.verify()
        with Image.open(io.BytesIO(payload)) as candidate:
            candidate.load()
            if candidate.width * candidate.height > MAX_IMAGE_PIXELS:
                raise UploadValidationError("图片像素尺寸过大，请压缩后重试。")
            return candidate.convert("RGB").copy()
    except UploadValidationError:
        raise
    except (OSError, UnidentifiedImageError, ValueError) as exc:
        raise UploadValidationError("无法读取该图片，请确认文件未损坏。") from exc


def low_confidence_message(low_confidence: bool) -> str | None:
    if low_confidence:
        return "模型把握较低，图片可能不属于训练分布。"
    return None


def load_analysis_artifacts(
    output_dir: Path,
    training_dir: Path | None = None,
    comparison_path: Path | None = None,
) -> AnalysisArtifacts:
    output_dir = Path(output_dir)
    training_dir = Path(training_dir) if training_dir is not None else output_dir
    comparison_path = (
        Path(comparison_path)
        if comparison_path is not None
        else output_dir / "comparison.csv"
    )
    metrics_path = output_dir / "metrics.json"
    training_curve = training_dir / "training_curves.png"
    confusion_matrix = output_dir / "confusion_matrix_normalized.png"
    errors_path = output_dir / "errors.csv"
    return AnalysisArtifacts(
        metrics=(
            json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics_path.is_file()
            else None
        ),
        comparison=pd.read_csv(comparison_path) if comparison_path.is_file() else None,
        training_curve=training_curve if training_curve.is_file() else None,
        confusion_matrix=confusion_matrix if confusion_matrix.is_file() else None,
        errors=pd.read_csv(errors_path) if errors_path.is_file() else None,
    )


@st.cache_resource(show_spinner="正在加载模型……", max_entries=3)
def cached_inference_bundle(checkpoint_path: str) -> InferenceBundle:
    return load_inference_bundle(Path(checkpoint_path))


def main() -> None:
    st.set_page_config(
        page_title="番茄叶片病害识别系统",
        page_icon=":material/eco:",
        layout="wide",
    )
    page = st.navigation(
        [
            st.Page("app_pages/home.py", title="首页", icon=":material/home:"),
            st.Page(
                "app_pages/recognition.py",
                title="叶片识别",
                icon=":material/image_search:",
            ),
            st.Page(
                "app_pages/analysis.py",
                title="模型分析",
                icon=":material/analytics:",
            ),
        ],
        position="top",
    )
    st.title(page.title, icon=page.icon)
    page.run()


if __name__ == "__main__":
    main()
