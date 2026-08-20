from __future__ import annotations

from collections.abc import Hashable, Sequence


def calculate_accuracy(predictions: Sequence[Hashable], targets: Sequence[Hashable]) -> float:
    """Return the proportion of predictions that equal their targets."""
    _validate_lengths(predictions, targets)
    if not targets:
        return 0.0
    return sum(prediction == target for prediction, target in zip(predictions, targets)) / len(targets)


def calculate_precision(
    predictions: Sequence[Hashable], targets: Sequence[Hashable], positive_label: Hashable
) -> float:
    """Return the positive predictive value for the explicit positive label."""
    true_positives, false_positives, _ = _binary_counts(predictions, targets, positive_label)
    denominator = true_positives + false_positives
    return true_positives / denominator if denominator else 0.0


def calculate_recall(
    predictions: Sequence[Hashable], targets: Sequence[Hashable], positive_label: Hashable
) -> float:
    """Return the fraction of positive targets identified by the predictions."""
    true_positives, _, false_negatives = _binary_counts(predictions, targets, positive_label)
    denominator = true_positives + false_negatives
    return true_positives / denominator if denominator else 0.0


def calculate_f1(
    predictions: Sequence[Hashable], targets: Sequence[Hashable], positive_label: Hashable
) -> float:
    """Return the harmonic mean of precision and recall for the positive label."""
    precision = calculate_precision(predictions, targets, positive_label)
    recall = calculate_recall(predictions, targets, positive_label)
    return 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0


def _binary_counts(
    predictions: Sequence[Hashable], targets: Sequence[Hashable], positive_label: Hashable
) -> tuple[int, int, int]:
    _validate_lengths(predictions, targets)
    true_positives = sum(
        prediction == positive_label and target == positive_label
        for prediction, target in zip(predictions, targets)
    )
    false_positives = sum(
        prediction == positive_label and target != positive_label
        for prediction, target in zip(predictions, targets)
    )
    false_negatives = sum(
        prediction != positive_label and target == positive_label
        for prediction, target in zip(predictions, targets)
    )
    return true_positives, false_positives, false_negatives


def _validate_lengths(predictions: Sequence[Hashable], targets: Sequence[Hashable]) -> None:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
