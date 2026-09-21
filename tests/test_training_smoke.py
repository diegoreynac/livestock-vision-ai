from pathlib import Path

import pytest
import torch

from src.models.torch_models import DualViewTorchModel
from src.training.checkpoint import CheckpointManager
from src.training.trainer import Trainer
from src.training.torch_dataset import InputMode


def synthetic_batch(input_mode: InputMode, batch_size: int = 2):
    """Create a small synthetic batch matching the training pipeline contract."""
    batch = {
        "weight": torch.tensor(
            [[150.0], [180.0]],
            dtype=torch.float32,
        ),
        "animal_id": ["animal_001", "animal_002"],
    }

    if input_mode is InputMode.SIDE:
        batch["image"] = torch.randn(batch_size, 3, 64, 64)

    elif input_mode is InputMode.REAR:
        batch["image"] = torch.randn(batch_size, 3, 64, 64)

    elif input_mode is InputMode.SIDE_REAR:
        batch["side_image"] = torch.randn(batch_size, 3, 64, 64)
        batch["rear_image"] = torch.randn(batch_size, 3, 64, 64)

    else:
        raise ValueError(f"Unsupported input mode: {input_mode}")

    return batch


def smoke_loss(outputs, batch):
    """Artificial loss used only to validate forward/backward integration."""
    return torch.mean((outputs.weight - batch["weight"]) ** 2)


@pytest.mark.parametrize(
    "input_mode",
    [
        InputMode.SIDE,
        InputMode.REAR,
        InputMode.SIDE_REAR,
    ],
)
def test_dual_view_model_trainer_smoke(
    input_mode: InputMode,
    tmp_path: Path,
):
    """Validate the complete training path with synthetic data."""

    model = DualViewTorchModel(
        architecture="mobilenet",
        variant="small",
        input_mode=input_mode,
        share_backbone=False,
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    checkpoint_manager = CheckpointManager(
        output_directory=tmp_path,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=smoke_loss,
        checkpoint_manager=checkpoint_manager,
        device="cpu",
    )

    batch = synthetic_batch(input_mode)

    result = trainer.train_epoch([batch])

    assert result.loss >= 0.0
    assert "mae" in result.metrics
    assert result.metrics["mae"] >= 0.0
    assert torch.isfinite(torch.tensor(result.metrics["mae"]))

    checkpoint_path = checkpoint_manager.save_last(
        epoch=1,
        model=model,
        optimizer=optimizer,
        validation_metric=result.loss,
    )

    assert checkpoint_path.exists()

    checkpoint = checkpoint_manager.load(
        checkpoint_path,
        model=model,
    )

    assert checkpoint["epoch"] == 1
    assert "model_state_dict" in checkpoint
    assert "optimizer_state_dict" in checkpoint