from __future__ import annotations

import unittest

from src.benchmarking.config import BenchmarkConfig


class BenchmarkConfigTests(unittest.TestCase):
    def test_defaults_are_valid(self) -> None:
        config = BenchmarkConfig()

        self.assertGreaterEqual(config.warmup_runs, 0)
        self.assertGreater(config.measurement_runs, 0)

    def test_negative_warmup_runs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(warmup_runs=-1)

    def test_non_positive_measurement_runs_are_rejected(self) -> None:
        with self.assertRaises(ValueError):
            BenchmarkConfig(measurement_runs=0)
