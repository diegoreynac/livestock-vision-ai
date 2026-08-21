"""Framework-independent inference latency measurement and statistics."""

from __future__ import annotations

from collections.abc import Callable
import statistics
import time

from src.benchmarking.config import BenchmarkConfig


def measure_latency(
    inference_function: Callable[[], object], config: BenchmarkConfig
) -> list[float]:
    """Execute warm-ups then return per-inference measured latencies in ms."""

    for _ in range(config.warmup_runs):
        inference_function()

    measurements: list[float] = []
    for _ in range(config.measurement_runs):
        start = time.perf_counter()
        inference_function()
        measurements.append((time.perf_counter() - start) * 1000)
    return measurements


def calculate_mean_latency(latencies_ms: list[float]) -> float | None:
    """Return arithmetic mean latency, or ``None`` when no measurements exist."""

    return statistics.fmean(latencies_ms) if latencies_ms else None


def calculate_median_latency(latencies_ms: list[float]) -> float | None:
    """Return median latency, or ``None`` when no measurements exist."""

    return statistics.median(latencies_ms) if latencies_ms else None


def calculate_p95_latency(latencies_ms: list[float]) -> float | None:
    """Return P95 using linear interpolation at index ``0.95 * (n - 1)``.

    This inclusive percentile convention returns the sole value for a
    one-element sample and interpolates between adjacent sorted observations.
    """

    if not latencies_ms:
        return None

    ordered = sorted(latencies_ms)
    position = 0.95 * (len(ordered) - 1)
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return ordered[lower_index] + fraction * (ordered[upper_index] - ordered[lower_index])


def calculate_fps(mean_latency_ms: float | None) -> float | None:
    """Return throughput in frames per second, or ``None`` for invalid latency."""

    if mean_latency_ms is None or mean_latency_ms <= 0:
        return None
    return 1000 / mean_latency_ms
