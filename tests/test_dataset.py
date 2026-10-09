import csv
import hashlib
from pathlib import Path

import pytest
import torch
from PIL import Image
from torch.utils.data import RandomSampler, SequentialSampler

from config import CLASS_IDS, DataConfig
from dataset import (
    ManifestMismatchError,
    TomatoDataset,
    build_dataloaders,
    build_transform,
)


def _pattern_image(path: Path, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    image = Image.new(mode, (48, 40))
    for x in range(image.width):
        for y in range(image.height):
            value = (x * 5 + y * 11) % 256
            if mode == "RGB":
                image.putpixel((x, y), (value, (value * 3) % 256, (value * 7) % 256))
            else:
                image.putpixel((x, y), value)
    image.save(path)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("relative_path", "class_id", "sha256", "split")
        )
        writer.writeheader()
        writer.writerows(rows)


def _three_split_fixture(tmp_path: Path) -> tuple[Path, Path]:
    data_root = tmp_path / "raw"
    manifest_path = tmp_path / "manifest.csv"
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        relative = Path(CLASS_IDS[index]) / f"{split}.png"
        image_path = data_root / relative
        _pattern_image(image_path, mode="L" if split == "test" else "RGB")
        rows.append(
            {
                "relative_path": relative.as_posix(),
                "class_id": CLASS_IDS[index],
                "sha256": _sha256(image_path),
                "split": split,
            }
        )
    _write_manifest(manifest_path, rows)
    return data_root, manifest_path


def test_dataset_uses_fixed_class_index_and_only_requested_split(tmp_path):
    data_root, manifest_path = _three_split_fixture(tmp_path)
    dataset = TomatoDataset(
        data_root, manifest_path, "val", build_transform(training=False)
    )

    tensor, class_index, relative_path = dataset[0]

    assert len(dataset) == 1
    assert class_index == 1
    assert relative_path.endswith("val.png")
    assert tensor.shape == (3, 224, 224)


def test_evaluation_transform_is_repeatable_and_converts_grayscale(tmp_path):
    data_root, manifest_path = _three_split_fixture(tmp_path)
    dataset = TomatoDataset(
        data_root, manifest_path, "test", build_transform(training=False)
    )

    first, _, _ = dataset[0]
    second, _, _ = dataset[0]

    assert first.shape == (3, 224, 224)
    assert torch.equal(first, second)


def test_training_transform_changes_a_patterned_image():
    image = Image.new("RGB", (48, 40))
    for x in range(48):
        for y in range(40):
            image.putpixel((x, y), ((x * 5) % 256, (y * 7) % 256, (x + y) % 256))
    transform = build_transform(training=True)

    samples = [transform(image) for _ in range(5)]

    assert all(sample.shape == (3, 224, 224) for sample in samples)
    assert any(not torch.equal(samples[0], sample) for sample in samples[1:])


@pytest.mark.parametrize("mutation", ["missing", "changed"])
def test_manifest_mismatch_is_detected_before_iteration(tmp_path, mutation):
    data_root, manifest_path = _three_split_fixture(tmp_path)
    target = data_root / CLASS_IDS[0] / "train.png"
    if mutation == "missing":
        target.unlink()
    else:
        target.write_bytes(target.read_bytes() + b"changed")

    with pytest.raises(ManifestMismatchError, match="train.png"):
        TomatoDataset(data_root, manifest_path, "train", build_transform(False))


def test_dataloader_factory_shuffles_only_training_split(tmp_path):
    data_root, manifest_path = _three_split_fixture(tmp_path)
    config = DataConfig(data_root=data_root, manifest_path=manifest_path)

    loaders = build_dataloaders(config, batch_size=2, num_workers=0)

    assert set(loaders) == {"train", "val", "test"}
    assert isinstance(loaders["train"].sampler, RandomSampler)
    assert isinstance(loaders["val"].sampler, SequentialSampler)
    assert isinstance(loaders["test"].sampler, SequentialSampler)
    assert all(loader.num_workers == 0 for loader in loaders.values())
