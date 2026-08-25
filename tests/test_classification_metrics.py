from __future__ import annotations

import unittest

from src.evaluation.classification import (
    calculate_accuracy,
    calculate_f1,
    calculate_precision,
    calculate_recall,
)


class TestClassificationMetrics(unittest.TestCase):
    def test_perfect_predictions(self) -> None:
        predictions = ["F", "M", "F", "M"]
        self.assertEqual(calculate_accuracy(predictions, predictions), 1.0)
        self.assertEqual(calculate_precision(predictions, predictions, "F"), 1.0)
        self.assertEqual(calculate_recall(predictions, predictions, "F"), 1.0)
        self.assertEqual(calculate_f1(predictions, predictions, "F"), 1.0)

    def test_mixed_predictions(self) -> None:
        predictions = ["F", "M", "M", "F"]
        targets = ["F", "F", "M", "M"]
        self.assertEqual(calculate_accuracy(predictions, targets), 0.5)
        self.assertEqual(calculate_precision(predictions, targets, "F"), 0.5)
        self.assertEqual(calculate_recall(predictions, targets, "F"), 0.5)

    def test_all_predictions_positive(self) -> None:
        self.assertEqual(calculate_precision(["M", "M"], ["M", "F"], "M"), 0.5)
        self.assertEqual(calculate_recall(["M", "M"], ["M", "F"], "M"), 1.0)

    def test_no_positive_predictions(self) -> None:
        predictions = ["F", "F"]
        targets = ["M", "F"]
        self.assertEqual(calculate_precision(predictions, targets, "M"), 0.0)
        self.assertEqual(calculate_recall(predictions, targets, "M"), 0.0)
        self.assertEqual(calculate_f1(predictions, targets, "M"), 0.0)

    def test_positive_label_is_explicit(self) -> None:
        predictions = ["F", "M", "M", "F"]
        targets = ["F", "F", "M", "M"]
        self.assertEqual(calculate_precision(predictions, targets, "M"), 0.5)
        self.assertEqual(calculate_recall(predictions, targets, "M"), 0.5)

    def test_f1_matches_precision_and_recall(self) -> None:
        predictions = ["F", "M", "F"]
        targets = ["F", "F", "M"]
        precision = calculate_precision(predictions, targets, "F")
        recall = calculate_recall(predictions, targets, "F")
        self.assertEqual(calculate_f1(predictions, targets, "F"), 2 * precision * recall / (precision + recall))


if __name__ == "__main__":
    unittest.main()
