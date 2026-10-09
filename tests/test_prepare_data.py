from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from config import CLASS_IDS
from prepare_data import (
    DuplicateLabelError,
    ImageRecord,
    InvalidImageError,
    discover_images,
    make_stratified_split,
    write_manifest,
)


def _write_image(path: Path, color: tuple[int, int, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color).save(path)


def _build_source(root: Path, samples_per_class: int = 12) -> None:
    for class_index, class_id in enumerate(CLASS_IDS):
        for sample_index in range(samples_per_class):
            color = (
                (class_index * 23 + sample_index * 7) % 256,
                (class_index * 41 + sample_index * 11) % 256,
                (class_index * 59 + sample_index * 13) % 256,
            )
            _write_image(root / class_id / f"sample-{sample_index:02d}.png", color)


def test_discover_images_uses_only_configured_classes(tmp_path):
    _build_source(tmp_path, samples_per_class=1)
    _write_image(tmp_path / "Apple___healthy" / "ignored.png", (1, 2, 3))

    records = discover_images(tmp_path, CLASS_IDS)

    assert len(records) == 10
    assert {record.class_id for record in records} == set(CLASS_IDS)
    assert all("Apple___healthy" not in record.relative_path for record in records)


def test_discover_images_reports_the_unreadable_relative_path(tmp_path):
    _build_source(tmp_path, samples_per_class=1)
    bad_path = tmp_path / CLASS_IDS[3] / "bad.jpg"
    bad_path.write_bytes(b"not an image")

    with pytest.raises(InvalidImageError, match=r"Tomato___Leaf_Mold[/\\]bad.jpg"):
        discover_images(tmp_path, CLASS_IDS)


def test_discover_images_rejects_missing_configured_class_directory(tmp_path):
    for class_id in CLASS_IDS[:-1]:
        _write_image(tmp_path / class_id / "sample.png", (1, 2, 3))

    with pytest.raises(FileNotFoundError, match="Tomato___healthy"):
        discover_images(tmp_path, CLASS_IDS)


def test_duplicate_content_is_kept_in_one_split(tmp_path):
    _build_source(tmp_path)
    original = tmp_path / CLASS_IDS[0] / "sample-00.png"
    duplicate = tmp_path / CLASS_IDS[0] / "copy.png"
    duplicate.write_bytes(original.read_bytes())
    records = discover_images(tmp_path, CLASS_IDS)

    split_records = make_stratified_split(records, (0.70, 0.15, 0.15), seed=42)
    matching = [record for record in split_records if record.sha256 == records[0].sha256]

    assert len(matching) == 2
    assert len({record.split for record in matching}) == 1


def test_grouped_split_uses_all_splits_when_group_sizes_make_it_feasible():
    records = []
    for group_index, group_size in enumerate((6, 2, 2)):
        records.extend(
            ImageRecord(
                relative_path=f"group-{group_index}/copy-{copy_index}.png",
                class_id=CLASS_IDS[0],
                sha256=f"hash-{group_index}",
            )
            for copy_index in range(group_size)
        )

    split = make_stratified_split(records, (0.70, 0.15, 0.15), seed=42)

    assert Counter(record.split for record in split) == {
        "train": 6,
        "val": 2,
        "test": 2,
    }


def test_every_class_reaches_all_splits_and_is_reproducible(tmp_path):
    _build_source(tmp_path)
    records = discover_images(tmp_path, CLASS_IDS)

    first = make_stratified_split(records, (0.70, 0.15, 0.15), seed=42)
    second = make_stratified_split(records, (0.70, 0.15, 0.15), seed=42)
    first_manifest = tmp_path / "first.csv"
    second_manifest = tmp_path / "second.csv"
    write_manifest(first, first_manifest)
    write_manifest(second, second_manifest)

    for class_id in CLASS_IDS:
        assert {record.split for record in first if record.class_id == class_id} == {
            "train",
            "val",
            "test",
        }
    assert first_manifest.read_bytes() == second_manifest.read_bytes()


def test_same_content_under_different_labels_is_rejected(tmp_path):
    _build_source(tmp_path, samples_per_class=1)
    source = tmp_path / CLASS_IDS[0] / "sample-00.png"
    target = tmp_path / CLASS_IDS[1] / "conflict.png"
    target.write_bytes(source.read_bytes())
    records = discover_images(tmp_path, CLASS_IDS)

    with pytest.raises(DuplicateLabelError, match="different class labels"):
        make_stratified_split(records, (0.70, 0.15, 0.15), seed=42)
