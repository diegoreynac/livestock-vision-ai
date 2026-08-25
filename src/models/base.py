from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

from src.models.output import ModelOutput


class BaseModel(ABC):
    """Abstract base class defining the common model interface for architectures.

    Implementations must remain framework-agnostic at this abstraction level.
    """

    @abstractmethod
    def forward(self, *inputs: Any, **kwargs: Any) -> ModelOutput:
        """Perform a single forward pass. Returns raw ModelOutput."""

    @abstractmethod
    def predict(self, *inputs: Any, **kwargs: Any) -> ModelOutput:
        """Run inference and return a ModelOutput suitable for downstream use."""

    @abstractmethod
    def count_parameters(self) -> int:
        """Return the total number of model parameters as an int."""

    @abstractmethod
    def model_size(self) -> float:
        """Return a numeric model size (for example, megabytes)."""

    @abstractmethod
    def export(self, destination: Path | str, **kwargs: Any) -> None:
        """Export model artifacts to the provided destination.

        This method is intentionally generic and must not assume a specific
        runtime or export format.
        """
