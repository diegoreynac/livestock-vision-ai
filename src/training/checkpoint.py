from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn


class CheckpointManager:
    """Manage model training checkpoints."""

    LAST_CHECKPOINT = "last.pt"
    BEST_CHECKPOINT = "best.pt"

    def __init__(self, output_directory: Path) -> None:
        """Initialize the checkpoint manager."""

        if not isinstance(output_directory, Path):
            raise TypeError("output_directory must be a pathlib.Path.")

        self.output_directory = output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_metric: float,
        filename: str,
    ) -> Path:
        """Save a training checkpoint."""

        if epoch < 0:
            raise ValueError("epoch must be non-negative.")

        if not filename.strip():
            raise ValueError("filename must not be empty.")

        checkpoint_path = self.output_directory / filename

        checkpoint: dict[str, Any] = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "validation_metric": validation_metric,
        }

        torch.save(checkpoint, checkpoint_path)

        return checkpoint_path

    def save_last(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_metric: float,
    ) -> Path:
        """Save the latest training checkpoint."""

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_metric=validation_metric,
            filename=self.LAST_CHECKPOINT,
        )

    def save_best(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_metric: float,
    ) -> Path:
        """Save the best validation checkpoint."""

        return self.save(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            validation_metric=validation_metric,
            filename=self.BEST_CHECKPOINT,
        )

    def load(
        self,
        filename: str,
        model: nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:
        """Load a checkpoint into a model and optionally an optimizer."""

        if not filename.strip():
            raise ValueError("filename must not be empty.")

        checkpoint_path = self.output_directory / filename

        if not checkpoint_path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {checkpoint_path}"
            )

        checkpoint = torch.load(
            checkpoint_path,
            map_location=map_location,
        )

        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a dictionary.")

        required_keys = {
            "epoch",
            "model_state_dict",
            "optimizer_state_dict",
            "validation_metric",
        }

        missing_keys = required_keys - checkpoint.keys()

        if missing_keys:
            raise ValueError(
                f"Checkpoint is missing required keys: {sorted(missing_keys)}"
            )

        model.load_state_dict(checkpoint["model_state_dict"])

        if optimizer is not None:
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        return checkpoint