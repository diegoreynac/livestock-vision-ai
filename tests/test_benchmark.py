from __future__ import annotations

import unittest

from src.benchmarking.benchmark import run_benchmark
from src.benchmarking.config import BenchmarkConfig
from src.benchmarking.hardware import HardwareInfo, get_hardware_info
from src.evaluation.metrics import ModelComplexity


class HardwareTests(unittest.TestCase):
    def test_get_hardware_info_returns_strings(self) -> None:
        info = get_hardware_info()

        self.assertIsInstance(info, HardwareInfo)
        self.assertTrue(info.operating_system)
        self.assertTrue(info.python_version)


class BenchmarkTests(unittest.TestCase):
    def test_run_benchmark_preserves_supplied_metadata(self) -> None:
        complexity = ModelComplexity(100, 80, 1.5)
        hardware = HardwareInfo("Test OS", "Test platform", "CPU", "machine", "3.10")
        config = BenchmarkConfig(warmup_runs=1, measurement_runs=2, batch_size=4, input_shape=(3, 64, 64))

        result = run_benchmark(
            lambda: None,
            config,
            complexity,
            hardware_info=hardware,
            model_name="test-model",
        )

        self.assertEqual(result.complexity, complexity)
        self.assertEqual(result.hardware_info, hardware)
        self.assertEqual(result.warmup_runs, 1)
        self.assertEqual(result.measurement_runs, 2)
        self.assertEqual(result.batch_size, 4)
        self.assertEqual(result.input_shape, (3, 64, 64))
        self.assertEqual(result.total_parameters, 100)
        self.assertIsNotNone(result.latency_mean_ms)
        self.assertIsNotNone(result.latency_median_ms)
        self.assertIsNotNone(result.latency_p95_ms)
        self.assertIsNotNone(result.fps)
