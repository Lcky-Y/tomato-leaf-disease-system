import pytest
import torch

from models import build_model
from models.resnet18 import build_resnet18, set_resnet_train_stage


@pytest.mark.parametrize("model_name", ["simple_cnn", "resnet18"])
def test_models_accept_224_images_and_return_ten_logits(model_name):
    model = build_model(model_name, num_classes=10, pretrained=False)
    model.eval()

    with torch.no_grad():
        output = model(torch.randn(2, 3, 224, 224))

    assert output.shape == (2, 10)
    assert torch.isfinite(output).all()


def test_head_stage_trains_only_the_classifier():
    model = build_resnet18(num_classes=10, pretrained=False)

    returned = set_resnet_train_stage(model, "head")
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}

    assert trainable_names
    assert all(name.startswith("fc.") for name in trainable_names)
    assert {id(param) for param in returned} == {
        id(param) for param in model.parameters() if param.requires_grad
    }


def test_layer4_stage_trains_last_residual_block_and_classifier():
    model = build_resnet18(num_classes=10, pretrained=False)

    set_resnet_train_stage(model, "layer4")
    trainable_names = {name for name, param in model.named_parameters() if param.requires_grad}

    assert any(name.startswith("layer4.") for name in trainable_names)
    assert any(name.startswith("fc.") for name in trainable_names)
    assert all(
        name.startswith("layer4.") or name.startswith("fc.")
        for name in trainable_names
    )


def test_unknown_model_name_lists_valid_choices():
    with pytest.raises(ValueError, match=r"simple_cnn.*resnet18"):
        build_model("unknown", num_classes=10, pretrained=False)


def test_unknown_resnet_stage_is_rejected():
    model = build_resnet18(num_classes=10, pretrained=False)

    with pytest.raises(ValueError, match=r"head.*layer4"):
        set_resnet_train_stage(model, "everything")
