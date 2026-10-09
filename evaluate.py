"""Independent test-set evaluation and model comparison."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from torch import nn
from torch.utils.data import DataLoader

from config import CLASS_IDS, DataConfig
from dataset import TomatoDataset, build_transform
from models import VALID_MODEL_NAMES, build_model
from utils.checkpoints import (
    CheckpointCompatibilityError,
    load_checkpoint,
    load_model_weights,
)
from utils.reporting import atomic_write_json


@dataclass(frozen=True)
class PredictionTable:
    paths: tuple[str, ...]
    true_indices: tuple[int, ...]
    predicted_indices: tuple[int, ...]
    confidences: tuple[float, ...]
    logits: np.ndarray
    elapsed_ms: float
    device: str


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def collect_predictions(
    model: nn.Module,
    loader: Iterable,
    device: torch.device,
) -> PredictionTable:
    model.eval()
    all_paths: list[str] = []
    all_true: list[int] = []
    all_predicted: list[int] = []
    all_confidences: list[float] = []
    all_logits: list[np.ndarray] = []
    elapsed_ms = 0.0
    warmed_up = False

    with torch.no_grad():
        for images, labels, paths in loader:
            images = images.to(device)
            if not warmed_up:
                model(images)
                _synchronize(device)
                warmed_up = True
            _synchronize(device)
            started = time.perf_counter()
            logits = model(images)
            _synchronize(device)
            elapsed_ms += (time.perf_counter() - started) * 1000

            probabilities = torch.softmax(logits, dim=1)
            confidences, predicted = probabilities.max(dim=1)
            all_paths.extend(str(path) for path in paths)
            all_true.extend(labels.cpu().tolist())
            all_predicted.extend(predicted.cpu().tolist())
            all_confidences.extend(confidences.cpu().tolist())
            all_logits.append(logits.cpu().numpy())

    if not all_logits:
        raise ValueError("test loader produced no samples")
    return PredictionTable(
        paths=tuple(all_paths),
        true_indices=tuple(all_true),
        predicted_indices=tuple(all_predicted),
        confidences=tuple(float(value) for value in all_confidences),
        logits=np.concatenate(all_logits, axis=0),
        elapsed_ms=elapsed_ms,
        device=str(device),
    )


def measure_single_image_latency(
    model: nn.Module,
    image_size: int,
    device: torch.device,
    warmup: int = 3,
    runs: int = 20,
) -> float:
    if warmup < 0 or runs < 1:
        raise ValueError("warmup must be non-negative and runs must be positive")
    model.eval()
    sample = torch.zeros(1, 3, image_size, image_size, device=device)
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        _synchronize(device)
        started = time.perf_counter()
        for _ in range(runs):
            model(sample)
        _synchronize(device)
    return (time.perf_counter() - started) * 1000 / runs


def test_split_identity(data_config: DataConfig) -> tuple[str, int]:
    with Path(data_config.manifest_path).open(
        "r", encoding="utf-8", newline=""
    ) as stream:
        reader = csv.DictReader(stream)
        required = {"relative_path", "class_id", "sha256", "split"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError("manifest is missing fields required for test identity")
        rows = sorted(
            (
                row["relative_path"],
                row["class_id"],
                row["sha256"],
            )
            for row in reader
            if row["split"] == "test"
        )
    digest = hashlib.sha256(
        json.dumps(rows, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest, len(rows)


def compute_classification_metrics(
    table: PredictionTable,
    class_ids: tuple[str, ...],
) -> dict[str, object]:
    labels = list(range(len(class_ids)))
    precision, recall, f1, _ = precision_recall_fscore_support(
        table.true_indices,
        table.predicted_indices,
        labels=labels,
        average="macro",
        zero_division=0,
    )
    raw_confusion = confusion_matrix(
        table.true_indices, table.predicted_indices, labels=labels
    )
    row_sums = raw_confusion.sum(axis=1, keepdims=True)
    normalized_confusion = np.divide(
        raw_confusion,
        row_sums,
        out=np.zeros_like(raw_confusion, dtype=float),
        where=row_sums != 0,
    )
    report = classification_report(
        table.true_indices,
        table.predicted_indices,
        labels=labels,
        target_names=list(class_ids),
        output_dict=True,
        zero_division=0,
    )
    return {
        "class_ids": list(class_ids),
        "accuracy": float(accuracy_score(table.true_indices, table.predicted_indices)),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
        "macro_f1": float(f1),
        "classification_report": report,
        "confusion_matrix": raw_confusion.tolist(),
        "confusion_matrix_normalized": normalized_confusion.tolist(),
    }


def build_test_loader(
    data_config: DataConfig,
    image_size: int,
    mean: Sequence[float],
    std: Sequence[float],
) -> DataLoader:
    dataset = TomatoDataset(
        data_config.data_root,
        data_config.manifest_path,
        "test",
        build_transform(False, image_size, mean=mean, std=std),
    )
    return DataLoader(
        dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )


def _plot_confusion(matrix: np.ndarray, class_ids: tuple[str, ...], path: Path) -> None:
    fig, axis = plt.subplots(figsize=(12, 10))
    sns.heatmap(matrix, cmap="Blues", ax=axis, xticklabels=class_ids, yticklabels=class_ids)
    axis.set_xlabel("Predicted")
    axis.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _write_classification_report(report: dict[str, object], output_dir: Path) -> None:
    atomic_write_json(output_dir / "classification_report.json", report)
    columns = ("label", "precision", "recall", "f1-score", "support")
    with (output_dir / "classification_report.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        for label, values in report.items():
            if not isinstance(values, dict):
                continue
            writer.writerow({"label": label, **{key: values.get(key, "") for key in columns[1:]}})


def _write_error_artifacts(
    table: PredictionTable,
    class_ids: tuple[str, ...],
    output_dir: Path,
) -> None:
    errors = [
        {
            "path": path,
            "true_class": class_ids[true_index],
            "predicted_class": class_ids[predicted_index],
            "confidence": confidence,
        }
        for path, true_index, predicted_index, confidence in zip(
            table.paths,
            table.true_indices,
            table.predicted_indices,
            table.confidences,
        )
        if true_index != predicted_index
    ]
    errors.sort(key=lambda row: (-float(row["confidence"]), str(row["path"])))
    with (output_dir / "errors.csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "true_class", "predicted_class", "confidence")
        )
        writer.writeheader()
        writer.writerows(errors)

    correct = [
        {
            "path": path,
            "class_id": class_ids[true_index],
            "confidence": confidence,
        }
        for path, true_index, predicted_index, confidence in zip(
            table.paths,
            table.true_indices,
            table.predicted_indices,
            table.confidences,
        )
        if true_index == predicted_index
    ]
    correct.sort(key=lambda row: (-float(row["confidence"]), str(row["path"])))
    with (output_dir / "correct_samples.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("path", "class_id", "confidence")
        )
        writer.writeheader()
        writer.writerows(correct)

    pair_counts: dict[tuple[str, str], int] = {}
    for error in errors:
        pair = (str(error["true_class"]), str(error["predicted_class"]))
        pair_counts[pair] = pair_counts.get(pair, 0) + 1
    pairs = sorted(
        (
            {"true_class": true, "predicted_class": predicted, "count": count}
            for (true, predicted), count in pair_counts.items()
        ),
        key=lambda row: (-int(row["count"]), str(row["true_class"]), str(row["predicted_class"])),
    )
    with (output_dir / "confusion_pairs.csv").open(
        "w", encoding="utf-8", newline=""
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("true_class", "predicted_class", "count")
        )
        writer.writeheader()
        writer.writerows(pairs)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def evaluate_checkpoint(
    checkpoint_path: Path,
    data_config: DataConfig,
    output_dir: Path,
    device: str = "auto",
) -> Path:
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(f"evaluation output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    resolved_device = _resolve_device(device)
    state_dict, metadata = load_checkpoint(
        checkpoint_path,
        expected_classes=CLASS_IDS,
        map_location=resolved_device,
    )
    if metadata.model_name not in VALID_MODEL_NAMES:
        raise CheckpointCompatibilityError(
            f"checkpoint declares unsupported model: {metadata.model_name}"
        )
    model = build_model(
        metadata.model_name, len(metadata.class_ids), pretrained=False
    ).to(resolved_device)
    load_model_weights(model, state_dict)
    mean = metadata.normalization["mean"]
    std = metadata.normalization["std"]
    loader = build_test_loader(data_config, metadata.image_size, mean, std)
    table = collect_predictions(model, loader, resolved_device)
    metrics = compute_classification_metrics(table, metadata.class_ids)
    split_fingerprint, test_sample_count = test_split_identity(data_config)

    if resolved_device.type == "cpu":
        cpu_model = model
    else:
        cpu_model = build_model(
            metadata.model_name, len(metadata.class_ids), pretrained=False
        ).to(torch.device("cpu"))
        load_model_weights(cpu_model, state_dict)
    cpu_latency = measure_single_image_latency(
        cpu_model,
        metadata.image_size,
        torch.device("cpu"),
    )
    gpu_latency = (
        measure_single_image_latency(model, metadata.image_size, resolved_device)
        if resolved_device.type == "cuda"
        else None
    )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "model_name": metadata.model_name,
        **metrics,
        "parameter_count": parameter_count,
        "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
        "latency_ms_per_image": cpu_latency,
        "cpu_latency_ms_per_image": cpu_latency,
        "gpu_latency_ms_per_image": gpu_latency,
        "evaluation_forward_ms": table.elapsed_ms,
        "device": table.device,
        "test_split_fingerprint": split_fingerprint,
        "test_sample_count": test_sample_count,
    }
    metrics_path = output_dir / "metrics.json"
    atomic_write_json(metrics_path, payload)
    _write_classification_report(metrics["classification_report"], output_dir)
    _plot_confusion(np.asarray(metrics["confusion_matrix"]), metadata.class_ids, output_dir / "confusion_matrix_raw.png")
    _plot_confusion(
        np.asarray(metrics["confusion_matrix_normalized"]),
        metadata.class_ids,
        output_dir / "confusion_matrix_normalized.png",
    )
    _write_error_artifacts(table, metadata.class_ids, output_dir)
    return metrics_path


COMPARISON_COLUMNS = (
    "model_name",
    "accuracy",
    "macro_precision",
    "macro_recall",
    "macro_f1",
    "parameter_count",
    "checkpoint_bytes",
    "latency_ms_per_image",
    "cpu_latency_ms_per_image",
    "gpu_latency_ms_per_image",
    "device",
    "test_split_fingerprint",
    "test_sample_count",
)


def compare_runs(metric_paths: Sequence[Path], output_csv: Path) -> None:
    rows = []
    identities: set[tuple[object, object]] = set()
    for path in metric_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        identity = (
            payload.get("test_split_fingerprint"),
            payload.get("test_sample_count"),
        )
        if None in identity:
            raise ValueError("metrics must identify the test split before comparison")
        identities.add(identity)
        rows.append({column: payload[column] for column in COMPARISON_COLUMNS})
    if len(identities) != 1:
        raise ValueError("models must use the same test split for comparison")
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=COMPARISON_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=DataConfig().manifest_path)
    parser.add_argument("--data-root", type=Path, default=DataConfig().data_root)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    metrics_path = evaluate_checkpoint(
        args.checkpoint,
        DataConfig(data_root=args.data_root, manifest_path=args.manifest),
        args.output_dir,
        args.device,
    )
    print(f"Metrics: {metrics_path}")


if __name__ == "__main__":
    main()
