import csv

import pytest
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

import train
from config import CLASS_IDS, DataConfig, TrainingConfig
from train import EpochMetrics, evaluate_one_epoch, train_one_epoch
from utils.checkpoints import load_checkpoint


class LogitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.tensor(0.0))

    def forward(self, inputs):
        return inputs + self.anchor * 0.0


class TinyStagedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer4 = nn.Linear(2, 2)
        self.fc = nn.Linear(2, 10)

    def forward(self, inputs):
        return self.fc(self.layer4(inputs))


def _logit_loader():
    logits = torch.tensor(
        [
            [4.0, 0.0, 0.0],
            [0.0, 4.0, 0.0],
            [0.0, 3.0, 2.0],
        ]
    )
    labels = torch.tensor([0, 1, 2])
    paths = ["a.png", "b.png", "c.png"]
    return DataLoader(list(zip(logits, labels, paths)), batch_size=2, shuffle=False), logits, labels


def test_train_epoch_uses_sample_weighted_loss_and_full_epoch_metrics():
    loader, logits, labels = _logit_loader()
    model = LogitModel()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

    metrics = train_one_epoch(
        model, loader, nn.CrossEntropyLoss(), optimizer, torch.device("cpu")
    )

    predictions = logits.argmax(dim=1).numpy()
    assert metrics.loss == pytest.approx(F.cross_entropy(logits, labels).item())
    assert metrics.accuracy == pytest.approx(2 / 3)
    assert metrics.macro_f1 == pytest.approx(
        f1_score(labels.numpy(), predictions, average="macro", zero_division=0)
    )


def test_evaluation_does_not_create_gradients():
    loader, _, _ = _logit_loader()
    model = LogitModel()

    evaluate_one_epoch(model, loader, nn.CrossEntropyLoss(), torch.device("cpu"))

    assert model.anchor.grad is None


def _patch_fit_dependencies(monkeypatch, evaluation_values, stage_events=None):
    tiny = TinyStagedModel()
    dummy_loader = [(torch.zeros(1, 2), torch.zeros(1, dtype=torch.long), ["x"])]
    monkeypatch.setattr(
        train,
        "build_dataloaders",
        lambda *args, **kwargs: {"train": dummy_loader, "val": dummy_loader, "test": dummy_loader},
    )
    monkeypatch.setattr(train, "build_model", lambda *args, **kwargs: tiny)
    def fake_train_epoch(model, loader, criterion, optimizer, device):
        optimizer.zero_grad(set_to_none=True)
        loss = sum(parameter.sum() * 0.0 for parameter in model.parameters() if parameter.requires_grad)
        loss.backward()
        optimizer.step()
        return EpochMetrics(1.0, 0.5, 0.4)

    monkeypatch.setattr(train, "train_one_epoch", fake_train_epoch)
    values = iter(evaluation_values)
    monkeypatch.setattr(train, "evaluate_one_epoch", lambda *args, **kwargs: next(values))
    if stage_events is not None:
        def set_stage(model, stage):
            stage_events.append(stage)
            for parameter in model.parameters():
                parameter.requires_grad = False
            modules = [model.fc] if stage == "head" else [model.layer4, model.fc]
            for module in modules:
                for parameter in module.parameters():
                    parameter.requires_grad = True
            return [parameter for parameter in model.parameters() if parameter.requires_grad]

        monkeypatch.setattr(train, "set_resnet_train_stage", set_stage)
    return tiny


def test_resnet_fit_runs_head_then_layer4_and_saves_best_epoch(tmp_path, monkeypatch):
    stages = []
    learning_rates = []
    _patch_fit_dependencies(
        monkeypatch,
        [EpochMetrics(0.9, 0.6, 0.60), EpochMetrics(0.7, 0.8, 0.80)],
        stages,
    )
    original_train = train.train_one_epoch

    def record_learning_rate(model, loader, criterion, optimizer, device):
        learning_rates.append(optimizer.param_groups[0]["lr"])
        return original_train(model, loader, criterion, optimizer, device)

    monkeypatch.setattr(train, "train_one_epoch", record_learning_rate)
    config = TrainingConfig(head_epochs=1, finetune_epochs=1)

    checkpoint = train.fit("resnet18", DataConfig(), config, tmp_path / "run")
    _, metadata = load_checkpoint(checkpoint)

    assert stages == ["head", "layer4"]
    assert learning_rates == pytest.approx([1e-3, 1e-4])
    assert metadata.best_epoch == 2
    assert metadata.best_val_metrics["macro_f1"] == pytest.approx(0.80)
    with (tmp_path / "run" / "history.csv").open(encoding="utf-8", newline="") as stream:
        history = list(csv.DictReader(stream))
    assert [row["phase"] for row in history] == ["head", "layer4"]


def test_best_checkpoint_survives_a_later_training_error(tmp_path, monkeypatch):
    _patch_fit_dependencies(
        monkeypatch,
        [EpochMetrics(0.8, 0.7, 0.75), RuntimeError("interrupted")],
    )
    values = iter([EpochMetrics(0.8, 0.7, 0.75), RuntimeError("interrupted")])

    def evaluate_then_interrupt(*args, **kwargs):
        value = next(values)
        if isinstance(value, Exception):
            raise value
        return value

    monkeypatch.setattr(train, "evaluate_one_epoch", evaluate_then_interrupt)
    config = TrainingConfig(simple_epochs=3, early_stopping_patience=2)
    run_dir = tmp_path / "run"

    with pytest.raises(RuntimeError, match="interrupted"):
        train.fit("simple_cnn", DataConfig(), config, run_dir)

    _, metadata = load_checkpoint(run_dir / "best_model.pth")
    assert metadata.best_epoch == 1
    assert metadata.best_val_metrics["macro_f1"] == pytest.approx(0.75)
