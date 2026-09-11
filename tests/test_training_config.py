from pathlib import Path

import pytest

from src.training.config import TrainingConfig


def test_default_configuration() -> None:
    config = TrainingConfig()

    assert config.seed == 42
    assert config.input_size == (224, 224)
    assert config.batch_size == 16
    assert config.epochs == 50
    assert config.learning_rate == 1e-3
    assert config.optimizer == "adamw"
    assert config.scheduler is None
    assert config.weight_decay == 1e-4
    assert config.device == "auto"
    assert config.output_directory == Path("output/checkpoints")


def test_custom_configuration() -> None:
    config = TrainingConfig(
        seed=123,
        input_size=(256, 256),
        batch_size=32,
        epochs=100,
        learning_rate=1e-4,
        optimizer="sgd",
        scheduler="cosine",
        weight_decay=1e-5,
        device="cuda",
        output_directory=Path("custom/checkpoints"),
    )

    assert config.seed == 123
    assert config.input_size == (256, 256)
    assert config.batch_size == 32
    assert config.epochs == 100
    assert config.learning_rate == 1e-4
    assert config.optimizer == "sgd"
    assert config.scheduler == "cosine"
    assert config.weight_decay == 1e-5
    assert config.device == "cuda"
    assert config.output_directory == Path("custom/checkpoints")


@pytest.mark.parametrize(
    "field,value",
    [
        ("seed", -1),
        ("batch_size", 0),
        ("epochs", 0),
        ("learning_rate", 0),
        ("weight_decay", -1e-4),
    ],
)
def test_invalid_numeric_values(field: str, value: int | float) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


@pytest.mark.parametrize(
    "input_size",
    [
        (0, 224),
        (224, 0),
        (-1, 224),
    ],
)
def test_invalid_input_size(input_size: tuple[int, int]) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(input_size=input_size)


def test_invalid_optimizer() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(optimizer=" ")


def test_invalid_scheduler() -> None:
    with pytest.raises(ValueError):
        TrainingConfig(scheduler=" ")


@pytest.mark.parametrize("device", ["tpu", "gpu", "cuda:0", ""])
def test_invalid_device(device: str) -> None:
    with pytest.raises(ValueError):
        TrainingConfig(device=device)


def test_output_directory_must_be_path() -> None:
    with pytest.raises(TypeError):
        TrainingConfig(output_directory="output/checkpoints")  # type: ignore[arg-type]