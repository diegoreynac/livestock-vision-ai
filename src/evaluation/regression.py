from __future__ import annotations

from collections.abc import Sequence
from math import sqrt


def calculate_mae(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Return mean absolute error in kilograms."""
    _validate_lengths(predictions, targets)
    if not targets:
        return 0.0
    return sum(abs(prediction - target) for prediction, target in zip(predictions, targets)) / len(targets)


def calculate_rmse(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Return root mean squared error in kilograms."""
    _validate_lengths(predictions, targets)
    if not targets:
        return 0.0
    mean_squared_error = sum(
        (prediction - target) ** 2 for prediction, target in zip(predictions, targets)
    ) / len(targets)
    return sqrt(mean_squared_error)


def calculate_r2(predictions: Sequence[float], targets: Sequence[float]) -> float:
    """Return R-squared, or 0.0 when targets have zero variance."""
    _validate_lengths(predictions, targets)
    if not targets:
        return 0.0
    target_mean = sum(targets) / len(targets)
    total_sum_of_squares = sum((target - target_mean) ** 2 for target in targets)
    if total_sum_of_squares == 0.0:
        return 0.0
    residual_sum_of_squares = sum(
        (target - prediction) ** 2 for prediction, target in zip(predictions, targets)
    )
    return 1.0 - residual_sum_of_squares / total_sum_of_squares


def _validate_lengths(predictions: Sequence[float], targets: Sequence[float]) -> None:
    if len(predictions) != len(targets):
        raise ValueError("predictions and targets must have equal length")
