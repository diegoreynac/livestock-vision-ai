from pathlib import Path

import pytest
import torch
from torch import nn

from src.training.checkpoint import CheckpointManager
from src.training.trainer import EpochResult, Trainer


class SimpleModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = nn.Linear(4, 1)

    def forward(self, image: torch.Tensor):
        return self.linear(image)


def loss_function(
    outputs: torch.Tensor,
    batch: dict,
) -> torch.Tensor:
    targets = batch["target"]

    return torch.mean((outputs - targets) ** 2)


def create_trainer(
    tmp_path: Path,
) -> Trainer:
    model = SimpleModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    checkpoint_manager = CheckpointManager(tmp_path)

    return Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_function,
        checkpoint_manager=checkpoint_manager,
    )


def create_dataloader() -> list[dict]:
    return [
        {
            "image": torch.tensor(
                [[1.0, 2.0, 3.0, 4.0]],
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                [[5.0]],
                dtype=torch.float32,
            ),
        },
        {
            "image": torch.tensor(
                [[2.0, 3.0, 4.0, 5.0]],
                dtype=torch.float32,
            ),
            "target": torch.tensor(
                [[6.0]],
                dtype=torch.float32,
            ),
        },
    ]


def test_train_epoch_returns_epoch_result(tmp_path: Path) -> None:
    trainer = create_trainer(tmp_path)

    result = trainer.train_epoch(create_dataloader())

    assert isinstance(result, EpochResult)
    assert result.loss >= 0.0


def test_train_epoch_updates_model_parameters(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    before = trainer.model.linear.weight.detach().clone()

    trainer.train_epoch(create_dataloader())

    after = trainer.model.linear.weight.detach().clone()

    assert not torch.equal(before, after)


def test_validate_epoch_does_not_update_parameters(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    before = trainer.model.linear.weight.detach().clone()

    trainer.validate_epoch(create_dataloader())

    after = trainer.model.linear.weight.detach().clone()

    assert torch.equal(before, after)


def test_fit_returns_history(tmp_path: Path) -> None:
    trainer = create_trainer(tmp_path)

    history = trainer.fit(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
        epochs=2,
    )

    assert len(history) == 2
    assert history[0]["epoch"] == 0
    assert history[1]["epoch"] == 1


def test_fit_creates_last_checkpoint(tmp_path: Path) -> None:
    trainer = create_trainer(tmp_path)

    trainer.fit(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
        epochs=1,
    )

    assert (tmp_path / "last.pt").exists()


def test_fit_creates_best_checkpoint(tmp_path: Path) -> None:
    trainer = create_trainer(tmp_path)

    trainer.fit(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
        epochs=1,
    )

    assert (tmp_path / "best.pt").exists()


def test_empty_dataloader_returns_zero_loss(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    result = trainer.train_epoch([])

    assert result.loss == 0.0
    assert result.metrics == {}


def test_rejects_invalid_epochs(tmp_path: Path) -> None:
    trainer = create_trainer(tmp_path)

    with pytest.raises(
        ValueError,
        match="epochs must be positive",
    ):
        trainer.fit(
            train_dataloader=create_dataloader(),
            validation_dataloader=create_dataloader(),
            epochs=0,
        )


def test_rejects_empty_checkpoint_metric(
    tmp_path: Path,
) -> None:
    trainer = create_trainer(tmp_path)

    with pytest.raises(
        ValueError,
        match="checkpoint_metric must not be empty",
    ):
        trainer.fit(
            train_dataloader=create_dataloader(),
            validation_dataloader=create_dataloader(),
            epochs=1,
            checkpoint_metric="",
        )


def test_trainer_can_run_without_checkpoint_manager(
    tmp_path: Path,
) -> None:
    model = SimpleModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_function,
        checkpoint_manager=None,
    )

    history = trainer.fit(
        train_dataloader=create_dataloader(),
        validation_dataloader=create_dataloader(),
        epochs=1,
    )

    assert len(history) == 1