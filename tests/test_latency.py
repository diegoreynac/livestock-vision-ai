from __future__ import annotations

import unittest

from src.benchmarking.config import BenchmarkConfig
from src.benchmarking.latency import (
    calculate_fps,
    calculate_mean_latency,
    calculate_median_latency,
    calculate_p95_latency,
    measure_latency,
)


class LatencyTests(unittest.TestCase):
    def test_warmups_and_measurements_are_executed(self) -> None:
        calls = 0

        def inference() -> None:
            nonlocal calls
            calls += 1

        measurements = measure_latency(inference, BenchmarkConfig(2, 3))

        self.assertEqual(calls, 5)
        self.assertEqual(len(measurements), 3)
        self.assertTrue(all(latency >= 0 for latency in measurements))

    def test_statistics(self) -> None:
        latencies = [1.0, 2.0, 3.0, 4.0, 5.0]

        self.assertEqual(calculate_mean_latency(latencies), 3.0)
        self.assertEqual(calculate_median_latency(latencies), 3.0)
        self.assertEqual(calculate_p95_latency(latencies), 4.8)
        self.assertEqual(calculate_fps(4.0), 250.0)

    def test_empty_or_invalid_statistics_are_safe(self) -> None:
        self.assertIsNone(calculate_mean_latency([]))
        self.assertIsNone(calculate_median_latency([]))
        self.assertIsNone(calculate_p95_latency([]))
        self.assertIsNone(calculate_fps(None))
        self.assertIsNone(calculate_fps(0.0))
