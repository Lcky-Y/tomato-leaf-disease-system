"""Training entry point for SimpleCNN and staged ResNet18 fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterable

import torch
from sklearn.metrics import f1_score
from torch import nn

from config import CLASS_IDS, DataConfig, PathsConfig, TrainingConfig
from dataset import IMAGENET_MEAN, IMAGENET_STD, build_dataloaders
from models import build_model
from models.resnet18 import set_resnet_train_stage
from utils.checkpoints import CheckpointMetadata, save_checkpoint
from utils.reporting import append_history, atomic_write_json, plot_training_curves
from utils.reproducibility import runtime_metadata, seed_everything


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    accuracy: float
    macro_f1: float


def _epoch_metrics(total_loss: float, labels: list[int], predictions: list[int]) -> EpochMetrics:
    if not labels:
        raise ValueError("data loader produced no samples")
    correct = sum(label == prediction for label, prediction in zip(labels, predictions))
    return EpochMetrics(
        loss=total_loss / len(labels),
        accuracy=correct / len(labels),
        macro_f1=float(
            f1_score(labels, predictions, average="macro", zero_division=0)
        ),
    )


def train_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> EpochMetrics:
    model.train()
    _set_frozen_modules_to_eval(model)
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    for images, labels, _paths in loader:
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * labels.size(0)
        labels_all.extend(labels.detach().cpu().tolist())
        predictions_all.extend(logits.detach().argmax(dim=1).cpu().tolist())
    return _epoch_metrics(total_loss, labels_all, predictions_all)


def _set_frozen_modules_to_eval(module: nn.Module) -> None:
    for child in module.children():
        child_parameters = list(child.parameters())
        if child_parameters and not any(
            parameter.requires_grad for parameter in child_parameters
        ):
            child.eval()
        else:
            _set_frozen_modules_to_eval(child)


def evaluate_one_epoch(
    model: nn.Module,
    loader: Iterable,
    criterion: nn.Module,
    device: torch.device,
) -> EpochMetrics:
    model.eval()
    total_loss = 0.0
    labels_all: list[int] = []
    predictions_all: list[int] = []
    with torch.no_grad():
        for images, labels, _paths in loader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * labels.size(0)
            labels_all.extend(labels.cpu().tolist())
            predictions_all.extend(logits.argmax(dim=1).cpu().tolist())
    return _epoch_metrics(total_loss, labels_all, predictions_all)


def _resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device)


def fit(
    model_name: str,
    data_config: DataConfig,
    training_config: TrainingConfig,
    run_dir: Path,
    device: str = "auto",
) -> Path:
    run_dir = Path(run_dir)
    history_path = run_dir / "history.csv"
    if history_path.exists():
        raise FileExistsError(f"run directory already contains history: {run_dir}")
    run_dir.mkdir(parents=True, exist_ok=True)

    seed_everything(data_config.seed)
    resolved_device = _resolve_device(device)
    loaders = build_dataloaders(
        data_config, training_config.batch_size, training_config.num_workers
    )
    model = build_model(
        model_name,
        num_classes=len(CLASS_IDS),
        pretrained=model_name == "resnet18",
    ).to(resolved_device)
    criterion = nn.CrossEntropyLoss()
    best_path = run_dir / "best_model.pth"

    atomic_write_json(
        run_dir / "config.json",
        {
            "model_name": model_name,
            "data": asdict(data_config),
            "training": asdict(training_config),
            "device": str(resolved_device),
        },
    )
    atomic_write_json(run_dir / "runtime.json", runtime_metadata())

    if model_name == "simple_cnn":
        phases = [("simple", training_config.simple_epochs, training_config.learning_rate)]
    elif model_name == "resnet18":
        phases = [
            ("head", training_config.head_epochs, training_config.learning_rate),
            (
                "layer4",
                training_config.finetune_epochs,
                training_config.finetune_learning_rate,
            ),
        ]
    else:
        raise ValueError("model_name must be 'simple_cnn' or 'resnet18'")

    best_f1 = float("-inf")
    global_epoch = 0
    for phase, epochs, learning_rate in phases:
        if epochs <= 0:
            continue
        if model_name == "resnet18":
            trainable = set_resnet_train_stage(model, phase)
        else:
            trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(
            trainable, lr=learning_rate, weight_decay=training_config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, epochs)
        )
        stale_epochs = 0

        for _ in range(epochs):
            global_epoch += 1
            current_lr = optimizer.param_groups[0]["lr"]
            try:
                train_metrics = train_one_epoch(
                    model, loaders["train"], criterion, optimizer, resolved_device
                )
                val_metrics = evaluate_one_epoch(
                    model, loaders["val"], criterion, resolved_device
                )
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "CUDA out of memory. Re-run with a smaller --batch-size; "
                        "the recorded configuration was not changed."
                    ) from exc
                raise

            append_history(
                history_path,
                {
                    "epoch": global_epoch,
                    "phase": phase,
                    "learning_rate": current_lr,
                    "train_loss": train_metrics.loss,
                    "train_accuracy": train_metrics.accuracy,
                    "train_macro_f1": train_metrics.macro_f1,
                    "val_loss": val_metrics.loss,
                    "val_accuracy": val_metrics.accuracy,
                    "val_macro_f1": val_metrics.macro_f1,
                },
            )

            if val_metrics.macro_f1 > best_f1:
                best_f1 = val_metrics.macro_f1
                stale_epochs = 0
                save_checkpoint(
                    best_path,
                    model,
                    CheckpointMetadata(
                        schema_version=1,
                        model_name=model_name,
                        class_ids=CLASS_IDS,
                        image_size=data_config.image_size,
                        normalization={
                            "mean": list(IMAGENET_MEAN),
                            "std": list(IMAGENET_STD),
                        },
                        training_config=asdict(training_config),
                        best_epoch=global_epoch,
                        best_val_metrics=asdict(val_metrics),
                    ),
                )
            else:
                stale_epochs += 1
            scheduler.step()
            if stale_epochs >= training_config.early_stopping_patience:
                break

    if not best_path.exists():
        raise RuntimeError("training produced no checkpoint")
    plot_training_curves(history_path, run_dir / "training_curves.png")
    return best_path


def _build_parser() -> argparse.ArgumentParser:
    defaults = TrainingConfig()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=("simple_cnn", "resnet18"), required=True)
    parser.add_argument("--manifest", type=Path, default=DataConfig().manifest_path)
    parser.add_argument("--data-root", type=Path, default=DataConfig().data_root)
    parser.add_argument("--simple-epochs", type=int, default=defaults.simple_epochs)
    parser.add_argument("--head-epochs", type=int, default=defaults.head_epochs)
    parser.add_argument("--finetune-epochs", type=int, default=defaults.finetune_epochs)
    parser.add_argument("--batch-size", type=int, default=defaults.batch_size)
    parser.add_argument("--num-workers", type=int, default=defaults.num_workers)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--run-name")
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    training = replace(
        TrainingConfig(),
        simple_epochs=args.simple_epochs,
        head_epochs=args.head_epochs,
        finetune_epochs=args.finetune_epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    data = DataConfig(data_root=args.data_root, manifest_path=args.manifest)
    run_name = args.run_name or f"{args.model}-{datetime.now():%Y%m%d-%H%M%S}"
    checkpoint = fit(
        args.model,
        data,
        training,
        PathsConfig().outputs_dir / run_name,
        device=args.device,
    )
    print(f"Best checkpoint: {checkpoint}")


if __name__ == "__main__":
    main()
