"""Shared configuration for the tomato leaf disease project."""

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent

CLASS_IDS: tuple[str, ...] = (
    "Tomato___Bacterial_spot",
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___Septoria_leaf_spot",
    "Tomato___Spider_mites Two-spotted_spider_mite",
    "Tomato___Target_Spot",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    "Tomato___Tomato_mosaic_virus",
    "Tomato___healthy",
)

CLASS_NAMES_CN: dict[str, str] = {
    "Tomato___Bacterial_spot": "细菌性斑点病",
    "Tomato___Early_blight": "早疫病",
    "Tomato___Late_blight": "晚疫病",
    "Tomato___Leaf_Mold": "叶霉病",
    "Tomato___Septoria_leaf_spot": "斑枯病",
    "Tomato___Spider_mites Two-spotted_spider_mite": "二斑叶螨",
    "Tomato___Target_Spot": "靶斑病",
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "黄化曲叶病毒病",
    "Tomato___Tomato_mosaic_virus": "花叶病毒病",
    "Tomato___healthy": "健康叶片",
}


@dataclass(frozen=True)
class PathsConfig:
    project_root: Path = PROJECT_ROOT
    data_root: Path = PROJECT_ROOT / "data" / "raw"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.csv"
    checkpoints_dir: Path = PROJECT_ROOT / "checkpoints"
    outputs_dir: Path = PROJECT_ROOT / "outputs"


@dataclass(frozen=True)
class DataConfig:
    data_root: Path = PROJECT_ROOT / "data" / "raw"
    manifest_path: Path = PROJECT_ROOT / "data" / "manifest.csv"
    image_size: int = 224
    split_ratios: tuple[float, float, float] = (0.70, 0.15, 0.15)
    seed: int = 42


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 0
    simple_epochs: int = 20
    head_epochs: int = 5
    finetune_epochs: int = 15
    learning_rate: float = 1e-3
    finetune_learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    early_stopping_patience: int = 5
