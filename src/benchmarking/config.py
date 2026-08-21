"""Configuration for architecture-independent benchmark runs."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class BenchmarkConfig:
    """Settings for a benchmark run.

    Final values should be set by the project's experimental protocol. The
    defaults are safe, modest values intended for local development and tests.
    """

    warmup_runs: int = 5
    measurement_runs: int = 30
    batch_size: int | None = None
    input_shape: tuple[int, ...] | None = None
    random_seed: int | None = None

    def __post_init__(self) -> None:
        if self.warmup_runs < 0:
            raise ValueError("warmup_runs must be greater than or equal to zero")
        if self.measurement_runs <= 0:
            raise ValueError("measurement_runs must be greater than zero")
