"""Manifest-backed datasets and transforms for tomato leaf images."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
from typing import Callable, Sequence

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from config import CLASS_IDS, DataConfig


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
VALID_SPLITS = {"train", "val", "test"}


class ManifestMismatchError(ValueError):
    """Raised when a manifest no longer matches files on disk."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_transform(
    training: bool,
    image_size: int = 224,
    mean: Sequence[float] = IMAGENET_MEAN,
    std: Sequence[float] = IMAGENET_STD,
) -> Callable:
    normalize = transforms.Normalize(tuple(mean), tuple(std))
    if training:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.80, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.ColorJitter(
                    brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02
                ),
                transforms.ToTensor(),
                normalize,
            ]
        )
    resize_size = round(image_size * 256 / 224)
    return transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )


class TomatoDataset(Dataset):
    def __init__(
        self,
        data_root: Path,
        manifest_path: Path,
        split: str,
        transform: Callable | None,
    ) -> None:
        if split not in VALID_SPLITS:
            raise ValueError(f"split must be one of {sorted(VALID_SPLITS)}")
        self.data_root = Path(data_root)
        self.transform = transform or build_transform(training=False)
        self._rows = self._load_rows(Path(manifest_path), split)
        self._class_to_index = {class_id: index for index, class_id in enumerate(CLASS_IDS)}
        self._validate_files()

    @staticmethod
    def _load_rows(manifest_path: Path, split: str) -> list[dict[str, str]]:
        required = {"relative_path", "class_id", "sha256", "split"}
        try:
            with manifest_path.open("r", encoding="utf-8", newline="") as stream:
                reader = csv.DictReader(stream)
                if not reader.fieldnames or not required.issubset(reader.fieldnames):
                    raise ManifestMismatchError(
                        f"Manifest is missing required columns: {manifest_path}"
                    )
                rows = [row for row in reader if row["split"] == split]
        except FileNotFoundError as exc:
            raise ManifestMismatchError(f"Manifest not found: {manifest_path}") from exc
        for row in rows:
            if row["class_id"] not in CLASS_IDS:
                raise ManifestMismatchError(
                    f"Unknown class in manifest: {row['class_id']}"
                )
        return rows

    def _validate_files(self) -> None:
        for row in self._rows:
            image_path = self.data_root / Path(row["relative_path"])
            if not image_path.is_file():
                raise ManifestMismatchError(f"Manifest file is missing: {row['relative_path']}")
            if _sha256_file(image_path) != row["sha256"]:
                raise ManifestMismatchError(
                    f"Manifest hash mismatch: {row['relative_path']}"
                )

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, str]:
        row = self._rows[index]
        image_path = self.data_root / Path(row["relative_path"])
        with Image.open(image_path) as image:
            rgb_image = image.convert("RGB")
            tensor = self.transform(rgb_image)
        return tensor, self._class_to_index[row["class_id"]], row["relative_path"]


def build_dataloaders(
    data_config: DataConfig,
    batch_size: int,
    num_workers: int,
) -> dict[str, DataLoader]:
    loaders: dict[str, DataLoader] = {}
    generator = torch.Generator().manual_seed(data_config.seed)
    for split in ("train", "val", "test"):
        dataset = TomatoDataset(
            data_config.data_root,
            data_config.manifest_path,
            split,
            build_transform(split == "train", data_config.image_size),
        )
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if split == "train" else None,
        )
    return loaders
