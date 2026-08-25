"""Framework-independent utilities for repeatable model benchmarking."""

from __future__ import annotations

from src.benchmarking.benchmark import BenchmarkResult, run_benchmark
from src.benchmarking.config import BenchmarkConfig
from src.benchmarking.hardware import HardwareInfo, get_hardware_info

__all__ = [
    "BenchmarkConfig",
    "BenchmarkResult",
    "HardwareInfo",
    "get_hardware_info",
    "run_benchmark",
]
