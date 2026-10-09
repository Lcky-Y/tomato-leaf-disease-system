"""Offline end-to-end smoke coverage for the complete CPU pipeline."""

from dataclasses import replace

import pytest
from PIL import Image, ImageDraw

from config import CLASS_IDS, DataConfig, TrainingConfig
from evaluate import compare_runs, evaluate_checkpoint
from predict import load_inference_bundle, predict_image
from prepare_data import discover_images, make_stratified_split, write_manifest
from train import fit
from utils.checkpoints import load_checkpoint


def _write_synthetic_dataset(data_root):
    for class_index, class_id in enumerate(CLASS_IDS):
        class_dir = data_root / class_id
        class_dir.mkdir(parents=True)
        for sample_index in range(7):
            image = Image.new(
                "RGB",
                (40, 40),
                (
                    (class_index * 23 + sample_index * 7) % 256,
                    (class_index * 11 + sample_index * 31) % 256,
                    (class_index * 41 + sample_index * 13) % 256,
                ),
            )
            draw = ImageDraw.Draw(image)
            draw.rectangle(
                (sample_index, class_index, 20 + sample_index, 20 + class_index),
                outline=(255, 255, 255),
                width=2,
            )
            image.save(class_dir / f"sample-{sample_index}.png")


@pytest.mark.slow
def test_complete_cpu_pipeline_writes_loadable_artifacts(tmp_path):
    data_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.csv"
    run_dir = tmp_path / "run"
    evaluation_dir = tmp_path / "evaluation"
    _write_synthetic_dataset(data_root)

    discovered = discover_images(data_root, CLASS_IDS)
    split_records = make_stratified_split(discovered, (0.70, 0.15, 0.15), seed=42)
    write_manifest(split_records, manifest_path)

    split_counts = {
        split: sum(record.split == split for record in split_records)
        for split in ("train", "val", "test")
    }
    assert split_counts == {"train": 50, "val": 10, "test": 10}

    data_config = DataConfig(
        data_root=data_root,
        manifest_path=manifest_path,
        image_size=32,
        seed=42,
    )
    training_config = replace(
        TrainingConfig(),
        batch_size=16,
        num_workers=0,
        simple_epochs=1,
        early_stopping_patience=2,
    )
    checkpoint_path = fit(
        "simple_cnn",
        data_config,
        training_config,
        run_dir,
        device="cpu",
    )

    assert checkpoint_path.is_file()
    assert {
        "best_model.pth",
        "config.json",
        "history.csv",
        "runtime.json",
        "training_curves.png",
    }.issubset(path.name for path in run_dir.iterdir())
    _state_dict, metadata = load_checkpoint(checkpoint_path, map_location="cpu")
    assert metadata.model_name == "simple_cnn"
    assert metadata.class_ids == CLASS_IDS
    assert metadata.image_size == 32

    metrics_path = evaluate_checkpoint(
        checkpoint_path,
        data_config,
        evaluation_dir,
        device="cpu",
    )
    assert metrics_path.is_file()
    assert {
        "metrics.json",
        "classification_report.json",
        "classification_report.csv",
        "confusion_matrix_raw.png",
        "confusion_matrix_normalized.png",
        "errors.csv",
        "confusion_pairs.csv",
    }.issubset(path.name for path in evaluation_dir.iterdir())

    comparison_path = tmp_path / "comparison.csv"
    compare_runs([metrics_path], comparison_path)
    assert comparison_path.is_file()

    bundle = load_inference_bundle(checkpoint_path, device="cpu")
    with Image.open(data_root / CLASS_IDS[0] / "sample-0.png") as image:
        result = predict_image(image, bundle, top_k=3)
    assert len(result.items) == 3
    assert all(item.class_id in CLASS_IDS for item in result.items)
    assert result.items[0].probability >= result.items[1].probability
    assert result.items[1].probability >= result.items[2].probability
