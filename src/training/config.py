from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class TrainingConfig:
    seed: int | None = None
    input_size: tuple[int, int] | None = None
    batch_size: int | None = None
    epochs: int | None = None
    learning_rate: float | None = None
    optimizer: str | None = None
    scheduler: str | None = None
    weight_decay: float | None = None
    device: str | None = None
    output_directory: Path | None = None
