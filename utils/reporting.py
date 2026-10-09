"""Atomic report writers and training-curve plotting."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HISTORY_COLUMNS = (
    "epoch",
    "phase",
    "learning_rate",
    "train_loss",
    "train_accuracy",
    "train_macro_f1",
    "val_loss",
    "val_accuracy",
    "val_macro_f1",
)


def atomic_write_json(path: Path, payload: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def append_history(path: Path, row: Mapping[str, object]) -> None:
    path = Path(path)
    missing = [column for column in HISTORY_COLUMNS if column not in row]
    if missing:
        raise ValueError(f"history row is missing columns: {', '.join(missing)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    needs_header = not path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=HISTORY_COLUMNS, extrasaction="ignore")
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


def plot_training_curves(history_path: Path, output_path: Path) -> None:
    with Path(history_path).open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError("training history is empty")

    epochs = [int(row["epoch"]) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(epochs, [float(row["train_loss"]) for row in rows], label="train")
    axes[0].plot(epochs, [float(row["val_loss"]) for row in rows], label="val")
    axes[0].set_title("Loss")
    axes[1].plot(
        epochs, [float(row["train_accuracy"]) for row in rows], label="train"
    )
    axes[1].plot(epochs, [float(row["val_accuracy"]) for row in rows], label="val")
    axes[1].set_title("Accuracy")
    axes[2].plot(
        epochs, [float(row["train_macro_f1"]) for row in rows], label="train"
    )
    axes[2].plot(
        epochs, [float(row["val_macro_f1"]) for row in rows], label="val"
    )
    axes[2].set_title("Macro-F1")
    for axis in axes:
        axis.set_xlabel("Epoch")
        axis.grid(alpha=0.25)
        axis.legend()
    fig.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
