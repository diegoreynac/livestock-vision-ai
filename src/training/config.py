from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    """Configuration for a reproducible model training run."""

    seed: int = 42
    input_size: tuple[int, int] = (224, 224)
    batch_size: int = 16
    epochs: int = 50
    learning_rate: float = 1e-3
    optimizer: str = "adamw"
    scheduler: str | None = None
    weight_decay: float = 1e-4
    device: str = "auto"
    output_directory: Path = Path("output/checkpoints")

    def __post_init__(self) -> None:
        """Validate training configuration values."""

        if self.seed < 0:
            raise ValueError("seed must be non-negative.")

        if len(self.input_size) != 2:
            raise ValueError("input_size must contain exactly two dimensions.")

        height, width = self.input_size

        if height <= 0 or width <= 0:
            raise ValueError("input_size dimensions must be positive.")

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive.")

        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")

        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive.")

        if self.weight_decay < 0:
            raise ValueError("weight_decay must be non-negative.")

        if not self.optimizer.strip():
            raise ValueError("optimizer must not be empty.")

        if self.scheduler is not None and not self.scheduler.strip():
            raise ValueError("scheduler must not be empty when provided.")

        if self.device not in {"auto", "cpu", "cuda"}:
            raise ValueError("device must be one of: auto, cpu, cuda.")

        if not isinstance(self.output_directory, Path):
            raise TypeError("output_directory must be a pathlib.Path.")