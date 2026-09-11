from pathlib import Path

import pytest
import torch
from torch import nn

from src.training.checkpoint import CheckpointManager


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 2)


def create_model_and_optimizer() -> tuple[
    SimpleModel,
    torch.optim.Optimizer,
]:
    model = SimpleModel()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    return model, optimizer


def test_creates_output_directory(tmp_path: Path) -> None:
    output_directory = tmp_path / "checkpoints"

    CheckpointManager(output_directory)

    assert output_directory.exists()
    assert output_directory.is_dir()


def test_rejects_non_path_output_directory() -> None:
    with pytest.raises(
        TypeError,
        match="output_directory must be a pathlib.Path",
    ):
        CheckpointManager("output/checkpoints")  # type: ignore[arg-type]


def test_save_creates_checkpoint(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=5,
        validation_metric=12.5,
        filename="epoch_005.pt",
    )

    assert checkpoint_path == tmp_path / "epoch_005.pt"
    assert checkpoint_path.exists()


def test_save_last_uses_expected_filename(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=10,
        validation_metric=8.5,
    )

    assert checkpoint_path.name == "last.pt"
    assert checkpoint_path.exists()


def test_save_best_uses_expected_filename(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    checkpoint_path = manager.save_best(
        model=model,
        optimizer=optimizer,
        epoch=12,
        validation_metric=7.2,
    )

    assert checkpoint_path.name == "best.pt"
    assert checkpoint_path.exists()


def test_saved_checkpoint_contains_required_state(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    manager.save_best(
        model=model,
        optimizer=optimizer,
        epoch=15,
        validation_metric=6.8,
    )

    checkpoint = torch.load(
        tmp_path / "best.pt",
        map_location="cpu",
    )

    assert checkpoint["epoch"] == 15
    assert checkpoint["validation_metric"] == pytest.approx(6.8)
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint


def test_load_restores_model_and_optimizer(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)

    model, optimizer = create_model_and_optimizer()

    original_weight = model.linear.weight.detach().clone()

    manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=20,
        validation_metric=5.5,
    )

    with torch.no_grad():
        model.linear.weight.add_(10.0)

    assert not torch.equal(
        model.linear.weight,
        original_weight,
    )

    checkpoint = manager.load(
        filename="last.pt",
        model=model,
        optimizer=optimizer,
    )

    assert torch.equal(
        model.linear.weight,
        original_weight,
    )
    assert checkpoint["epoch"] == 20
    assert checkpoint["validation_metric"] == pytest.approx(5.5)


def test_load_can_restore_model_without_optimizer(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    manager.save_last(
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_metric=10.0,
    )

    new_model, _ = create_model_and_optimizer()

    checkpoint = manager.load(
        filename="last.pt",
        model=new_model,
    )

    assert checkpoint["epoch"] == 3


def test_load_rejects_missing_checkpoint(
    tmp_path: Path,
) -> None:
    manager = CheckpointManager(tmp_path)
    model, _ = create_model_and_optimizer()

    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        manager.load(
            filename="missing.pt",
            model=model,
        )


def test_save_rejects_negative_epoch(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    with pytest.raises(ValueError, match="epoch must be non-negative"):
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=-1,
            validation_metric=10.0,
            filename="invalid.pt",
        )


def test_save_rejects_empty_filename(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, optimizer = create_model_and_optimizer()

    with pytest.raises(ValueError, match="filename must not be empty"):
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=1,
            validation_metric=10.0,
            filename="",
        )


def test_load_rejects_empty_filename(tmp_path: Path) -> None:
    manager = CheckpointManager(tmp_path)
    model, _ = create_model_and_optimizer()

    with pytest.raises(ValueError, match="filename must not be empty"):
        manager.load(
            filename="",
            model=model,
        )