from __future__ import annotations

from collections.abc import Sequence

from src.evaluation.regression import calculate_mae
from src.evaluation.detection import (
    BoundingBox,
    DetectionPrediction,
    calculate_map,
)


class TrainingMetrics:
    """Accumulate and compute metrics during a training or validation epoch."""

    def __init__(self) -> None:
        """Initialize an empty metric accumulator."""

        self._weight_predictions: list[float] = []
        self._weight_targets: list[float] = []

        self._bbox_predictions: dict[str, list[DetectionPrediction]] = {}
        self._bbox_targets: dict[str, list[BoundingBox]] = {}

    def reset(self) -> None:
        """Clear all accumulated predictions and targets."""

        self._weight_predictions.clear()
        self._weight_targets.clear()

        self._bbox_predictions.clear()
        self._bbox_targets.clear()

    def update_weight(
        self,
        predictions: Sequence[float],
        targets: Sequence[float],
    ) -> None:
        """Add weight predictions and targets from one batch."""

        if len(predictions) != len(targets):
            raise ValueError(
                "weight predictions and targets must have equal length"
            )

        self._weight_predictions.extend(predictions)
        self._weight_targets.extend(targets)

    def update_bbox(
        self,
        image_ids: Sequence[str],
        predictions: Sequence[DetectionPrediction],
        targets: Sequence[BoundingBox],
    ) -> None:
        """Add bounding-box predictions and targets from one batch."""

        if len(image_ids) != len(predictions):
            raise ValueError(
                "image_ids and predictions must have equal length"
            )

        if len(image_ids) != len(targets):
            raise ValueError(
                "image_ids and targets must have equal length"
            )

        for image_id, prediction, target in zip(
            image_ids,
            predictions,
            targets,
        ):
            self._bbox_predictions.setdefault(
                image_id,
                [],
            ).append(prediction)

            self._bbox_targets.setdefault(
                image_id,
                [],
            ).append(target)

    def compute(self) -> dict[str, float]:
        """Compute all accumulated training metrics."""

        metrics: dict[str, float] = {}

        if self._weight_targets:
            metrics["mae"] = calculate_mae(
                self._weight_predictions,
                self._weight_targets,
            )

        if self._bbox_targets:
            metrics["map"] = calculate_map(
                self._bbox_predictions,
                self._bbox_targets,
            )

        return metrics