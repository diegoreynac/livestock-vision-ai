from __future__ import annotations

from typing import Iterable

import torch
from torch import nn

from src.training.config import TrainingConfig


class OptimizerFactory:
    """Create PyTorch optimizers from training configuration."""

    _SUPPORTED_OPTIMIZERS = {"adamw", "adam"}

    @classmethod
    def create(
        cls,
        model: nn.Module,
        config: TrainingConfig,
    ) -> torch.optim.Optimizer:
        """Create an optimizer configured for the given model."""

        if model is None:
            raise ValueError("model must not be None.")

        optimizer_name = config.optimizer.strip().lower()

        if optimizer_name not in cls._SUPPORTED_OPTIMIZERS:
            raise ValueError(
                f"Unsupported optimizer: {config.optimizer}. "
                f"Supported optimizers: {sorted(cls._SUPPORTED_OPTIMIZERS)}."
            )

        parameters = cls._trainable_parameters(model)

        if optimizer_name == "adamw":
            return torch.optim.AdamW(
                parameters,
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        return torch.optim.Adam(
            parameters,
            lr=config.learning_rate,
            weight_decay=config.weight_decay,
        )

    @staticmethod
    def _trainable_parameters(
        model: nn.Module,
    ) -> Iterable[nn.Parameter]:
        """Return trainable model parameters and reject empty parameter sets."""

        parameters = [
            parameter
            for parameter in model.parameters()
            if parameter.requires_grad
        ]

        if not parameters:
            raise ValueError("model must contain trainable parameters.")

        return parameters