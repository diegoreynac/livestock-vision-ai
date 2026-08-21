"""Assembly of framework-independent benchmark results."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from src.benchmarking.config import BenchmarkConfig
from src.benchmarking.hardware import HardwareInfo, get_hardware_info
from src.benchmarking.latency import (
    calculate_fps,
    calculate_mean_latency,
    calculate_median_latency,
    calculate_p95_latency,
    measure_latency,
)
from src.evaluation.metrics import InferencePerformance, ModelComplexity


@dataclass(slots=True)
class BenchmarkResult:
    """Benchmark outputs and the metadata needed to reproduce a run."""

    complexity: ModelComplexity
    performance: InferencePerformance
    latency_median_ms: float | None
    hardware_info: HardwareInfo
    warmup_runs: int
    measurement_runs: int
    batch_size: int | None = None
    input_shape: tuple[int, ...] | None = None
    model_name: str | None = None
    architecture_name: str | None = None

    @property
    def total_parameters(self) -> int | None:
        """Total parameters supplied by the architecture-specific caller."""

        return self.complexity.total_parameters

    @property
    def trainable_parameters(self) -> int | None:
        """Trainable parameters supplied by the architecture-specific caller."""

        return self.complexity.trainable_parameters

    @property
    def model_size_mb(self) -> float | None:
        """Model size supplied by the architecture-specific caller."""

        return self.complexity.model_size_mb

    @property
    def latency_mean_ms(self) -> float | None:
        """Arithmetic mean measured inference latency in milliseconds."""

        return self.performance.latency_ms

    @property
    def latency_p95_ms(self) -> float | None:
        """P95 measured inference latency in milliseconds."""

        return self.performance.p95_latency_ms

    @property
    def fps(self) -> float | None:
        """Measured throughput in frames per second."""

        return self.performance.fps


def run_benchmark(
    inference_function: Callable[[], object],
    config: BenchmarkConfig,
    complexity: ModelComplexity | None = None,
    *,
    hardware_info: HardwareInfo | None = None,
    model_name: str | None = None,
    architecture_name: str | None = None,
) -> BenchmarkResult:
    """Benchmark a supplied inference callable without assuming an ML framework."""

    latencies_ms = measure_latency(inference_function, config)
    mean_latency_ms = calculate_mean_latency(latencies_ms)
    resolved_hardware = hardware_info or get_hardware_info()
    performance = InferencePerformance(
        latency_ms=mean_latency_ms,
        p95_latency_ms=calculate_p95_latency(latencies_ms),
        fps=calculate_fps(mean_latency_ms),
        hardware=resolved_hardware.platform,
    )
    return BenchmarkResult(
        complexity=complexity or ModelComplexity(),
        performance=performance,
        latency_median_ms=calculate_median_latency(latencies_ms),
        hardware_info=resolved_hardware,
        warmup_runs=config.warmup_runs,
        measurement_runs=config.measurement_runs,
        batch_size=config.batch_size,
        input_shape=config.input_shape,
        model_name=model_name,
        architecture_name=architecture_name,
    )
