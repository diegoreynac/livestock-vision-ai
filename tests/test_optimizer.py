import pytest
import torch
from torch import nn

from src.training.config import TrainingConfig
from src.training.optimizer import OptimizerFactory


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


class FrozenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)

        for parameter in self.parameters():
            parameter.requires_grad = False


def test_creates_adamw_optimizer() -> None:
    model = SimpleModel()
    config = TrainingConfig(
        optimizer="adamw",
        learning_rate=2e-4,
        weight_decay=5e-5,
    )

    optimizer = OptimizerFactory.create(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(2e-4)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(5e-5)


def test_creates_adam_optimizer() -> None:
    model = SimpleModel()
    config = TrainingConfig(
        optimizer="adam",
        learning_rate=1e-3,
        weight_decay=1e-4,
    )

    optimizer = OptimizerFactory.create(model, config)

    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-3)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(1e-4)


def test_optimizer_name_is_case_insensitive() -> None:
    model = SimpleModel()
    config = TrainingConfig(optimizer="AdamW")

    optimizer = OptimizerFactory.create(model, config)

    assert isinstance(optimizer, torch.optim.AdamW)


def test_rejects_unsupported_optimizer() -> None:
    model = SimpleModel()
    config = TrainingConfig(optimizer="sgd")

    with pytest.raises(ValueError, match="Unsupported optimizer"):
        OptimizerFactory.create(model, config)


def test_rejects_none_model() -> None:
    config = TrainingConfig()

    with pytest.raises(ValueError, match="model must not be None"):
        OptimizerFactory.create(None, config)  # type: ignore[arg-type]


def test_rejects_model_without_trainable_parameters() -> None:
    model = FrozenModel()
    config = TrainingConfig()

    with pytest.raises(
        ValueError,
        match="model must contain trainable parameters",
    ):
        OptimizerFactory.create(model, config)


def test_only_trainable_parameters_are_registered() -> None:
    model = SimpleModel()
    model.linear.bias.requires_grad = False

    config = TrainingConfig()
    optimizer = OptimizerFactory.create(model, config)

    registered_parameters = list(optimizer.param_groups[0]["params"])

    assert registered_parameters == [model.linear.weight]