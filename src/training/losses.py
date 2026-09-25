from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class MultiTaskLossConfig:
    bbox_weight: float = 1.0
    weight_weight: float = 1.0
    bbox_loss: str = "smooth_l1"
    weight_loss: str = "smooth_l1"
    smooth_l1_beta: float = 1.0

    def __post_init__(self) -> None:
        for name, value in (("bbox_weight", self.bbox_weight), ("weight_weight", self.weight_weight), ("smooth_l1_beta", self.smooth_l1_beta)):
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                raise TypeError(f"{name} must be a real number.")
            if not torch.isfinite(torch.tensor(float(value))):
                raise ValueError(f"{name} must be finite.")
        if self.bbox_weight < 0.0 or self.weight_weight < 0.0:
            raise ValueError("Loss weights must be non-negative.")
        if self.bbox_weight == 0.0 and self.weight_weight == 0.0:
            raise ValueError("At least one loss weight must be positive.")
        if self.smooth_l1_beta <= 0.0:
            raise ValueError("smooth_l1_beta must be positive.")
        for name, value in (("bbox_loss", self.bbox_loss), ("weight_loss", self.weight_loss)):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must not be empty.")
            if value.lower() not in {"l1", "smooth_l1", "mse"}:
                raise ValueError(f"Unsupported {name} {value!r}; expected one of: l1, smooth_l1, mse.")


class MultiTaskLoss(nn.Module):
    """Differentiable weighted loss for BBox and weight regression."""

    def __init__(self, config: MultiTaskLossConfig | None = None) -> None:
        super().__init__()
        self.config = config if config is not None else MultiTaskLossConfig()

    def forward(self, outputs: Any, batch: dict[str, Any]) -> torch.Tensor:
        components: list[torch.Tensor] = []
        if self.config.bbox_weight > 0.0:
            bbox_losses = self._bbox_losses(outputs, batch)
            components.append(self.config.bbox_weight * torch.stack(bbox_losses).mean())
        if self.config.weight_weight > 0.0:
            prediction = getattr(outputs, "weight", None)
            target = batch.get("weight")
            if prediction is None:
                raise ValueError("Model output is missing weight prediction.")
            if target is None:
                raise ValueError("Batch is missing weight target.")
            self._validate_pair("weight", prediction, target, expected_last_dim=1)
            components.append(self.config.weight_weight * self._regression_loss(prediction, target, self.config.weight_loss))
        if not components:
            raise RuntimeError("No active loss components are configured.")
        return torch.stack(components).sum()

    def _bbox_losses(self, outputs: Any, batch: dict[str, Any]) -> list[torch.Tensor]:
        prediction_side = getattr(outputs, "bbox_side", None)
        prediction_rear = getattr(outputs, "bbox_rear", None)
        losses: list[torch.Tensor] = []
        if "bbox" in batch:
            prediction = prediction_side if prediction_side is not None else prediction_rear
            if prediction is None:
                raise ValueError("Model output is missing the BBox prediction.")
            target = batch["bbox"]
            self._validate_pair("bbox", prediction, target, expected_last_dim=4)
            return [self._regression_loss(prediction, target, self.config.bbox_loss)]
        found_view = False
        for view in ("side", "rear"):
            target = batch.get(f"bbox_{view}")
            prediction = getattr(outputs, f"bbox_{view}", None)
            if target is None and prediction is None:
                continue
            found_view = True
            if prediction is None:
                raise ValueError(f"Model output is missing the {view} BBox prediction.")
            if target is None:
                raise ValueError(f"Batch is missing the {view} BBox target.")
            self._validate_pair(f"bbox_{view}", prediction, target, expected_last_dim=4)
            losses.append(self._regression_loss(prediction, target, self.config.bbox_loss))
        if not found_view:
            raise ValueError("Batch is missing BBox targets.")
        return losses

    def _regression_loss(self, prediction: torch.Tensor, target: torch.Tensor, loss_name: str) -> torch.Tensor:
        name = loss_name.lower()
        if name == "l1":
            return F.l1_loss(prediction, target)
        if name == "mse":
            return F.mse_loss(prediction, target)
        return F.smooth_l1_loss(prediction, target, beta=self.config.smooth_l1_beta)

    @staticmethod
    def _validate_pair(name: str, prediction: torch.Tensor, target: torch.Tensor, *, expected_last_dim: int) -> None:
        if not isinstance(prediction, torch.Tensor):
            raise TypeError(f"{name} prediction must be a torch.Tensor.")
        if not isinstance(target, torch.Tensor):
            raise TypeError(f"{name} target must be a torch.Tensor.")
        if prediction.shape != target.shape:
            raise ValueError(f"{name} prediction and target shapes must match; got {tuple(prediction.shape)} and {tuple(target.shape)}.")
        if prediction.ndim != 2 or prediction.shape[-1] != expected_last_dim:
            raise ValueError(f"{name} tensors must have shape (B, {expected_last_dim}); got {tuple(prediction.shape)}.")
        if prediction.device != target.device:
            raise ValueError(f"{name} prediction and target must be on the same device.")


__all__ = ["MultiTaskLoss", "MultiTaskLossConfig"]
