import csv
import json

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader

import evaluate
from config import CLASS_IDS, DataConfig
from evaluate import (
    PredictionTable,
    collect_predictions,
    compare_runs,
    compute_classification_metrics,
    evaluate_checkpoint,
    measure_single_image_latency,
)
from models.simple_cnn import SimpleCNN
from utils.checkpoints import CheckpointMetadata, save_checkpoint


class IdentityLogitModel(nn.Module):
    def forward(self, inputs):
        return inputs


def _table() -> PredictionTable:
    logits = np.array(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 3.0, 2.0],
            [0.0, 5.0, 0.0],
        ],
        dtype=np.float32,
    )
    probabilities = torch.softmax(torch.from_numpy(logits), dim=1).numpy()
    return PredictionTable(
        paths=("correct-0.png", "wrong-0.png", "correct-1.png", "wrong-2.png"),
        true_indices=(0, 0, 1, 2),
        predicted_indices=(0, 1, 1, 1),
        confidences=tuple(float(value) for value in probabilities.max(axis=1)),
        logits=logits,
        elapsed_ms=8.0,
        device="cpu",
    )


def test_classification_metrics_have_fixed_class_order_and_exact_values():
    metrics = compute_classification_metrics(_table(), ("a", "b", "c"))

    assert metrics["class_ids"] == ["a", "b", "c"]
    assert metrics["accuracy"] == pytest.approx(0.5)
    assert metrics["macro_precision"] == pytest.approx(4 / 9)
    assert metrics["macro_recall"] == pytest.approx(0.5)
    assert metrics["macro_f1"] == pytest.approx(7 / 18)
    assert metrics["confusion_matrix"] == [[1, 1, 0], [0, 1, 0], [0, 1, 0]]


def test_collect_predictions_retains_paths_confidence_logits_and_cpu_timing():
    logits = torch.tensor([[4.0, 0.0], [0.0, 3.0], [0.0, 2.0]])
    labels = torch.tensor([0, 1, 0])
    loader = DataLoader(
        list(zip(logits, labels, ["a.png", "b.png", "error.png"])),
        batch_size=2,
        shuffle=False,
    )

    table = collect_predictions(IdentityLogitModel(), loader, torch.device("cpu"))

    assert table.paths == ("a.png", "b.png", "error.png")
    assert table.true_indices == (0, 1, 0)
    assert table.predicted_indices == (0, 1, 1)
    assert table.logits.shape == (3, 2)
    assert all(0.5 < confidence <= 1.0 for confidence in table.confidences)
    assert table.elapsed_ms >= 0.0
    assert table.device == "cpu"


def test_single_image_latency_uses_batch_size_one():
    seen_shapes = []

    class RecordingModel(nn.Module):
        def forward(self, inputs):
            seen_shapes.append(tuple(inputs.shape))
            return torch.zeros(inputs.size(0), 10)

    latency = measure_single_image_latency(
        RecordingModel(), image_size=32, device=torch.device("cpu"), warmup=1, runs=2
    )

    assert seen_shapes == [(1, 3, 32, 32)] * 3
    assert latency >= 0


def _checkpoint(path):
    metadata = CheckpointMetadata(
        schema_version=1,
        model_name="simple_cnn",
        class_ids=CLASS_IDS,
        image_size=224,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        training_config={"batch_size": 32},
        best_epoch=3,
        best_val_metrics={"macro_f1": 0.7},
    )
    save_checkpoint(path, SimpleCNN(10), metadata)


def test_evaluator_uses_only_test_loader_and_writes_complete_artifacts(
    tmp_path, monkeypatch
):
    checkpoint = tmp_path / "model.pth"
    _checkpoint(checkpoint)
    requested_splits = []

    def fake_test_loader(data_config, image_size, mean, std):
        requested_splits.append("test")
        return object()

    full_logits = np.pad(_table().logits, ((0, 0), (0, 7)))
    table = PredictionTable(
        paths=_table().paths,
        true_indices=_table().true_indices,
        predicted_indices=_table().predicted_indices,
        confidences=_table().confidences,
        logits=full_logits,
        elapsed_ms=8.0,
        device="cpu",
    )
    monkeypatch.setattr(evaluate, "build_test_loader", fake_test_loader)
    monkeypatch.setattr(evaluate, "collect_predictions", lambda *args: table)
    monkeypatch.setattr(
        evaluate, "measure_single_image_latency", lambda *args, **kwargs: 4.5
    )
    monkeypatch.setattr(
        evaluate, "test_split_identity", lambda *args: ("fingerprint", 4)
    )
    output_dir = tmp_path / "evaluation"

    metrics_path = evaluate_checkpoint(
        checkpoint, DataConfig(), output_dir, device="cpu"
    )

    assert requested_splits == ["test"]
    expected_files = {
        "metrics.json",
        "classification_report.json",
        "classification_report.csv",
        "confusion_matrix_raw.png",
        "confusion_matrix_normalized.png",
        "errors.csv",
        "correct_samples.csv",
        "confusion_pairs.csv",
    }
    assert expected_files <= {path.name for path in output_dir.iterdir()}
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert metrics["model_name"] == "simple_cnn"
    assert metrics["class_ids"] == list(CLASS_IDS)
    assert metrics["parameter_count"] > 0
    assert metrics["checkpoint_bytes"] == checkpoint.stat().st_size
    assert metrics["device"] == "cpu"
    assert metrics["cpu_latency_ms_per_image"] == pytest.approx(4.5)
    assert metrics["latency_ms_per_image"] == pytest.approx(4.5)
    assert metrics["test_split_fingerprint"] == "fingerprint"
    assert metrics["test_sample_count"] == 4
    with (output_dir / "errors.csv").open(encoding="utf-8", newline="") as stream:
        errors = list(csv.DictReader(stream))
    assert [row["path"] for row in errors] == ["wrong-2.png", "wrong-0.png"]
    with (output_dir / "correct_samples.csv").open(
        encoding="utf-8", newline=""
    ) as stream:
        correct = list(csv.DictReader(stream))
    assert [row["path"] for row in correct] == ["correct-0.png", "correct-1.png"]

    with pytest.raises(FileExistsError, match="not empty"):
        evaluate_checkpoint(checkpoint, DataConfig(), output_dir, device="cpu")


def test_compare_runs_writes_deterministic_summary(tmp_path):
    first = tmp_path / "a.json"
    second = tmp_path / "b.json"
    first.write_text(
        json.dumps(
            {
                "model_name": "simple_cnn",
                "accuracy": 0.7,
                "macro_precision": 0.6,
                "macro_recall": 0.5,
                "macro_f1": 0.55,
                "parameter_count": 10,
                "checkpoint_bytes": 20,
                "latency_ms_per_image": 2.0,
                "cpu_latency_ms_per_image": 2.0,
                "gpu_latency_ms_per_image": None,
                "device": "cpu",
                "test_split_fingerprint": "same-split",
                "test_sample_count": 10,
            }
        ),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps(
            {
                "model_name": "resnet18",
                "accuracy": 0.8,
                "macro_precision": 0.7,
                "macro_recall": 0.6,
                "macro_f1": 0.65,
                "parameter_count": 30,
                "checkpoint_bytes": 40,
                "latency_ms_per_image": 3.0,
                "cpu_latency_ms_per_image": 3.0,
                "gpu_latency_ms_per_image": 1.0,
                "device": "cpu",
                "test_split_fingerprint": "same-split",
                "test_sample_count": 10,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "comparison.csv"

    compare_runs([second, first], output)

    with output.open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert [row["model_name"] for row in rows] == ["resnet18", "simple_cnn"]
    assert rows[0]["cpu_latency_ms_per_image"] == "3.0"
    assert rows[0]["gpu_latency_ms_per_image"] == "1.0"


def test_compare_runs_rejects_different_test_splits(tmp_path):
    payload = {
        "model_name": "simple_cnn",
        "accuracy": 0.7,
        "macro_precision": 0.6,
        "macro_recall": 0.5,
        "macro_f1": 0.55,
        "parameter_count": 10,
        "checkpoint_bytes": 20,
        "latency_ms_per_image": 2.0,
        "cpu_latency_ms_per_image": 2.0,
        "gpu_latency_ms_per_image": None,
        "device": "cpu",
        "test_sample_count": 10,
    }
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first.write_text(
        json.dumps({**payload, "test_split_fingerprint": "split-a"}),
        encoding="utf-8",
    )
    second.write_text(
        json.dumps({**payload, "test_split_fingerprint": "split-b"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="same test split"):
        compare_runs([first, second], tmp_path / "comparison.csv")


def test_evaluator_rejects_reversed_project_class_order(tmp_path):
    checkpoint = tmp_path / "reversed.pth"
    metadata = CheckpointMetadata(
        schema_version=1,
        model_name="simple_cnn",
        class_ids=tuple(reversed(CLASS_IDS)),
        image_size=224,
        normalization={
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225],
        },
        training_config={},
        best_epoch=1,
        best_val_metrics={"macro_f1": 0.1},
    )
    save_checkpoint(checkpoint, SimpleCNN(10), metadata)

    from utils.checkpoints import CheckpointCompatibilityError

    with pytest.raises(CheckpointCompatibilityError, match="class order"):
        evaluate_checkpoint(
            checkpoint, DataConfig(), tmp_path / "evaluation", device="cpu"
        )
