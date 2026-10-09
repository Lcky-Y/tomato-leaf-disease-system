"""Independent test-set evaluation and model comparison."""

from __future__ import annotations

import argparse
import csv
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

from config import DataConfig
from dataset import TomatoDataset, build_transform
from models import build_model
from utils.checkpoints import load_checkpoint
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
    state_dict, metadata = load_checkpoint(checkpoint_path, map_location=resolved_device)
    model = build_model(
        metadata.model_name, len(metadata.class_ids), pretrained=False
    ).to(resolved_device)
    model.load_state_dict(state_dict)
    mean = metadata.normalization["mean"]
    std = metadata.normalization["std"]
    loader = build_test_loader(data_config, metadata.image_size, mean, std)
    table = collect_predictions(model, loader, resolved_device)
    metrics = compute_classification_metrics(table, metadata.class_ids)

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    payload = {
        "model_name": metadata.model_name,
        **metrics,
        "parameter_count": parameter_count,
        "checkpoint_bytes": Path(checkpoint_path).stat().st_size,
        "latency_ms_per_image": table.elapsed_ms / len(table.paths),
        "device": table.device,
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
    "device",
)


def compare_runs(metric_paths: Sequence[Path], output_csv: Path) -> None:
    rows = []
    for path in metric_paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        rows.append({column: payload[column] for column in COMPARISON_COLUMNS})
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
