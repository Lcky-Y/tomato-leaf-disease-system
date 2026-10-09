import csv
import json

from utils.reporting import HISTORY_COLUMNS, append_history, atomic_write_json, plot_training_curves


def _row(epoch: int, val_macro_f1: float) -> dict[str, object]:
    return {
        "epoch": epoch,
        "phase": "simple",
        "learning_rate": 0.001,
        "train_loss": 1.0 / epoch,
        "train_accuracy": 0.5,
        "train_macro_f1": 0.4,
        "val_loss": 1.2 / epoch,
        "val_accuracy": 0.6,
        "val_macro_f1": val_macro_f1,
    }


def test_atomic_json_write_replaces_destination_without_temp_file(tmp_path):
    path = tmp_path / "nested" / "config.json"

    atomic_write_json(path, {"seed": 42, "name": "番茄"})

    assert json.loads(path.read_text(encoding="utf-8")) == {
        "seed": 42,
        "name": "番茄",
    }
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_append_history_writes_one_header_and_fixed_columns(tmp_path):
    path = tmp_path / "history.csv"

    append_history(path, _row(1, 0.5))
    append_history(path, _row(2, 0.6))

    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert rows[0].keys() == dict.fromkeys(HISTORY_COLUMNS).keys()
    assert [int(row["epoch"]) for row in rows] == [1, 2]


def test_plot_training_curves_creates_nonempty_png(tmp_path):
    history = tmp_path / "history.csv"
    output = tmp_path / "curves.png"
    append_history(history, _row(1, 0.5))
    append_history(history, _row(2, 0.6))

    plot_training_curves(history, output)

    assert output.is_file()
    assert output.stat().st_size > 0
