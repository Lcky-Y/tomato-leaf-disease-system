"""Validate PlantVillage tomato images and create a reproducible split manifest."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from PIL import Image, UnidentifiedImageError

from config import CLASS_IDS, DataConfig


SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLIT_NAMES = ("train", "val", "test")


class InvalidImageError(ValueError):
    """Raised when an image cannot be decoded safely."""


class DuplicateLabelError(ValueError):
    """Raised when identical bytes are assigned to multiple labels."""


@dataclass(frozen=True)
class ImageRecord:
    relative_path: str
    class_id: str
    sha256: str


@dataclass(frozen=True)
class SplitRecord:
    relative_path: str
    class_id: str
    sha256: str
    split: str


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def discover_images(source_dir: Path, class_ids: tuple[str, ...]) -> list[ImageRecord]:
    source_dir = Path(source_dir)
    records: list[ImageRecord] = []
    for class_id in class_ids:
        class_dir = source_dir / class_id
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing configured class directory: {class_id}")
        paths = sorted(
            path
            for path in class_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
        )
        for path in paths:
            relative_path = path.relative_to(source_dir).as_posix()
            try:
                with Image.open(path) as image:
                    image.verify()
            except (OSError, UnidentifiedImageError, ValueError) as exc:
                raise InvalidImageError(f"Invalid image: {relative_path}") from exc
            records.append(ImageRecord(relative_path, class_id, _sha256_file(path)))
    return records


def _target_counts(total: int, ratios: tuple[float, float, float]) -> list[int]:
    raw = [total * ratio for ratio in ratios]
    counts = [math.floor(value) for value in raw]
    for index in sorted(range(3), key=lambda i: raw[i] - counts[i], reverse=True)[
        : total - sum(counts)
    ]:
        counts[index] += 1
    return counts


def make_stratified_split(
    records: Sequence[ImageRecord],
    ratios: tuple[float, float, float],
    seed: int,
) -> list[SplitRecord]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios):
        raise ValueError("ratios must contain three non-negative values")
    if not math.isclose(sum(ratios), 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("ratios must sum to 1.0")

    hash_labels: dict[str, str] = {}
    grouped: dict[str, dict[str, list[ImageRecord]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for record in records:
        prior_label = hash_labels.setdefault(record.sha256, record.class_id)
        if prior_label != record.class_id:
            raise DuplicateLabelError(
                f"Identical content has different class labels: {prior_label} and {record.class_id}"
            )
        grouped[record.class_id][record.sha256].append(record)

    rng = random.Random(seed)
    result: list[SplitRecord] = []
    for class_id in sorted(grouped):
        groups = [
            sorted(group, key=lambda item: item.relative_path)
            for _, group in sorted(grouped[class_id].items())
        ]
        rng.shuffle(groups)
        targets = _target_counts(sum(len(group) for group in groups), ratios)
        counts = [0, 0, 0]

        for group in groups:
            split_index = max(
                range(3),
                key=lambda index: (
                    targets[index] - counts[index],
                    ratios[index],
                    -index,
                ),
            )
            split_name = SPLIT_NAMES[split_index]
            counts[split_index] += len(group)
            result.extend(
                SplitRecord(item.relative_path, item.class_id, item.sha256, split_name)
                for item in group
            )

    return sorted(
        result,
        key=lambda item: (item.class_id, SPLIT_NAMES.index(item.split), item.relative_path),
    )


def write_manifest(records: Sequence[SplitRecord], destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=("relative_path", "class_id", "sha256", "split"),
        )
        writer.writeheader()
        for record in records:
            writer.writerow(
                {
                    "relative_path": record.relative_path,
                    "class_id": record.class_id,
                    "sha256": record.sha256,
                    "split": record.split,
                }
            )


def _print_summary(records: Sequence[SplitRecord]) -> None:
    counts = Counter((record.split, record.class_id) for record in records)
    print(f"Prepared {len(records)} images")
    for split in SPLIT_NAMES:
        total = sum(counts[(split, class_id)] for class_id in CLASS_IDS)
        print(f"{split}: {total}")
        for class_id in CLASS_IDS:
            print(f"  {class_id}: {counts[(split, class_id)]}")


def main() -> None:
    defaults = DataConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=defaults.data_root)
    parser.add_argument("--manifest", type=Path, default=defaults.manifest_path)
    args = parser.parse_args()

    records = discover_images(args.source, CLASS_IDS)
    split_records = make_stratified_split(
        records, defaults.split_ratios, defaults.seed
    )
    write_manifest(split_records, args.manifest)
    _print_summary(split_records)


if __name__ == "__main__":
    main()
