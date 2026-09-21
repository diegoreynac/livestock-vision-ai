from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from src.training.checkpoint import CheckpointManager
from src.training.metrics import TrainingMetrics


LossFunction = Callable[[Any, dict[str, Any]], torch.Tensor]


@dataclass(slots=True)
class EpochResult:
    """Results produced by one training or validation epoch."""

    loss: float
    metrics: dict[str, float]


class Trainer:
    """Coordinate model training, validation, metrics, and checkpoints."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: LossFunction,
        checkpoint_manager: CheckpointManager | None = None,
        device: torch.device | str = "cpu",
    ) -> None:
        """Initialize the trainer."""

        if model is None:
            raise ValueError("model must not be None.")

        if optimizer is None:
            raise ValueError("optimizer must not be None.")

        if loss_fn is None:
            raise ValueError("loss_fn must not be None.")

        self.model = model
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.checkpoint_manager = checkpoint_manager

        self.device = torch.device(device)
        self.model.to(self.device)

    def train_epoch(
        self,
        dataloader: Iterable[dict[str, Any]],
    ) -> EpochResult:
        """Run one training epoch."""

        self.model.train()

        metrics = TrainingMetrics()
        total_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            self.optimizer.zero_grad()

            outputs = self.model(**self._model_inputs(batch))
            loss = self.loss_fn(outputs, batch)

            if not isinstance(loss, torch.Tensor):
                raise TypeError("loss_fn must return a torch.Tensor.")

            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor.")

            loss.backward()
            self.optimizer.step()

            total_loss += loss.detach().item()
            batch_count += 1

            self._update_metrics(metrics, outputs, batch)

        return EpochResult(
            loss=self._average_loss(total_loss, batch_count),
            metrics=metrics.compute(),
        )

    @torch.no_grad()
    def validate_epoch(
        self,
        dataloader: Iterable[dict[str, Any]],
    ) -> EpochResult:
        """Run one validation epoch without updating model parameters."""

        self.model.eval()

        metrics = TrainingMetrics()
        total_loss = 0.0
        batch_count = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            outputs = self.model(**self._model_inputs(batch))
            loss = self.loss_fn(outputs, batch)

            if not isinstance(loss, torch.Tensor):
                raise TypeError("loss_fn must return a torch.Tensor.")

            if loss.ndim != 0:
                raise ValueError("loss_fn must return a scalar tensor.")

            total_loss += loss.detach().item()
            batch_count += 1

            self._update_metrics(metrics, outputs, batch)

        return EpochResult(
            loss=self._average_loss(total_loss, batch_count),
            metrics=metrics.compute(),
        )

    def fit(
        self,
        train_dataloader: Iterable[dict[str, Any]],
        validation_dataloader: Iterable[dict[str, Any]],
        epochs: int,
        checkpoint_metric: str = "mae",
    ) -> list[dict[str, Any]]:
        """Train for multiple epochs and save the best checkpoint."""

        if epochs <= 0:
            raise ValueError("epochs must be positive.")

        if not checkpoint_metric.strip():
            raise ValueError("checkpoint_metric must not be empty.")

        history: list[dict[str, Any]] = []
        best_metric = float("inf")

        for epoch in range(epochs):
            train_result = self.train_epoch(train_dataloader)
            validation_result = self.validate_epoch(validation_dataloader)

            validation_metric = validation_result.metrics.get(
                checkpoint_metric,
                validation_result.loss,
            )

            if self.checkpoint_manager is not None:
                self.checkpoint_manager.save_last(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    validation_metric=validation_metric,
                )

                if validation_metric < best_metric:
                    best_metric = validation_metric

                    self.checkpoint_manager.save_best(
                        model=self.model,
                        optimizer=self.optimizer,
                        epoch=epoch,
                        validation_metric=validation_metric,
                    )

            history.append(
                {
                    "epoch": epoch,
                    "train_loss": train_result.loss,
                    "train_metrics": train_result.metrics,
                    "validation_loss": validation_result.loss,
                    "validation_metrics": validation_result.metrics,
                }
            )

        return history

    def _model_inputs(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Extract and adapt batch inputs to the model's input contract."""

        input_mode = getattr(self.model, "input_mode", None)

        if "image" in batch:
            if input_mode is None:
                return {"image": batch["image"]}

            mode_name = getattr(input_mode, "name", str(input_mode))

            if mode_name == "SIDE":
                return {"side": batch["image"]}

            if mode_name == "REAR":
                return {"rear": batch["image"]}

            raise ValueError(
                f"Batch contains a single 'image', but model input mode is {mode_name}."
            )

        if "side_image" in batch and "rear_image" in batch:
            return {
                "side": batch["side_image"],
                "rear": batch["rear_image"],
            }

        raise ValueError("Batch does not contain model input images.")

    def _move_batch_to_device(
        self,
        batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Move tensor values in a batch to the configured device."""

        return {
            key: value.to(self.device)
            if isinstance(value, torch.Tensor)
            else value
            for key, value in batch.items()
        }

    @staticmethod
    def _average_loss(
        total_loss: float,
        batch_count: int,
    ) -> float:
        """Return average loss and handle empty dataloaders."""

        if batch_count == 0:
            return 0.0

        return total_loss / batch_count

    @staticmethod
    def _update_metrics(
        metrics: TrainingMetrics,
        outputs: Any,
        batch: dict[str, Any],
    ) -> None:
        """Update available metrics from model outputs and batch targets."""

        if hasattr(outputs, "weight") and outputs.weight is not None:
            weight_predictions = outputs.weight.detach().cpu().reshape(-1).tolist()

            weight_targets = (
                batch["weight"]
                .detach()
                .cpu()
                .reshape(-1)
                .tolist()
            )

            metrics.update_weight(
                predictions=weight_predictions,
                targets=weight_targets,
            )