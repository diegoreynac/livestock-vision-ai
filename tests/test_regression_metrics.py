from __future__ import annotations

import unittest

from src.evaluation.regression import calculate_mae, calculate_r2, calculate_rmse


class TestRegressionMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        values = [100.0, 120.0, 140.0]
        self.assertEqual(calculate_mae(values, values), 0.0)
        self.assertEqual(calculate_rmse(values, values), 0.0)
        self.assertEqual(calculate_r2(values, values), 1.0)

    def test_known_absolute_errors(self) -> None:
        self.assertEqual(calculate_mae([101.0, 118.0, 143.0], [100.0, 120.0, 140.0]), 2.0)

    def test_known_squared_errors_and_rmse(self) -> None:
        self.assertAlmostEqual(calculate_rmse([102.0, 116.0], [100.0, 120.0]), 10.0**0.5)

    def test_known_r2(self) -> None:
        self.assertAlmostEqual(calculate_r2([2.0, 4.0, 6.0], [1.0, 3.0, 5.0]), 0.625)

    def test_constant_targets_return_zero_r2(self) -> None:
        self.assertEqual(calculate_r2([100.0, 100.0], [100.0, 100.0]), 0.0)


if __name__ == "__main__":
    unittest.main()

