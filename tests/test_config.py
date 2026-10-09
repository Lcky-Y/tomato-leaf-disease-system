from dataclasses import FrozenInstanceError

import pytest


EXPECTED_CLASS_IDS = (
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


def test_class_ids_are_the_fixed_ten_tomato_labels():
    from config import CLASS_IDS

    assert CLASS_IDS == EXPECTED_CLASS_IDS
    assert len(CLASS_IDS) == len(set(CLASS_IDS)) == 10
    assert all(class_id.startswith("Tomato___") for class_id in CLASS_IDS)


def test_data_defaults_are_reproducible():
    from config import DataConfig

    config = DataConfig()

    assert config.seed == 42
    assert config.image_size == 224
    assert config.split_ratios == (0.70, 0.15, 0.15)
    assert sum(config.split_ratios) == pytest.approx(1.0)


def test_training_defaults_match_the_approved_plan():
    from config import TrainingConfig

    config = TrainingConfig()

    assert config.batch_size == 32
    assert config.num_workers == 0
    assert config.simple_epochs == 20
    assert config.head_epochs == 5
    assert config.finetune_epochs == 15
    assert config.learning_rate == pytest.approx(1e-3)
    assert config.finetune_learning_rate == pytest.approx(1e-4)
    assert config.weight_decay == pytest.approx(1e-4)
    assert config.early_stopping_patience == 5


def test_configuration_objects_are_frozen():
    from config import DataConfig

    config = DataConfig()

    with pytest.raises(FrozenInstanceError):
        config.seed = 7

