import pytest

from src.evaluation.detection import DetectionPrediction
from src.training.metrics import TrainingMetrics


def test_initial_metrics_are_empty() -> None:
    metrics = TrainingMetrics()

    assert metrics.compute() == {}


def test_calculates_weight_mae() -> None:
    metrics = TrainingMetrics()

    metrics.update_weight(
        predictions=[150.0, 180.0, 200.0],
        targets=[160.0, 175.0, 190.0],
    )

    result = metrics.compute()

    assert result["mae"] == pytest.approx(8.3333333333)


def test_accumulates_weight_predictions_across_batches() -> None:
    metrics = TrainingMetrics()

    metrics.update_weight(
        predictions=[150.0, 180.0],
        targets=[160.0, 175.0],
    )

    metrics.update_weight(
        predictions=[200.0],
        targets=[190.0],
    )

    result = metrics.compute()

    assert result["mae"] == pytest.approx(8.3333333333)


def test_rejects_mismatched_weight_lengths() -> None:
    metrics = TrainingMetrics()

    with pytest.raises(
        ValueError,
        match="weight predictions and targets must have equal length",
    ):
        metrics.update_weight(
            predictions=[150.0, 180.0],
            targets=[160.0],
        )


def test_calculates_bbox_map() -> None:
    metrics = TrainingMetrics()

    metrics.update_bbox(
        image_ids=["animal_001"],
        predictions=[
            DetectionPrediction(
                bbox=(10.0, 10.0, 50.0, 50.0),
                confidence=0.95,
            )
        ],
        targets=[
            (10.0, 10.0, 50.0, 50.0),
        ],
    )

    result = metrics.compute()

    assert result["map"] == pytest.approx(1.0)


def test_accumulates_bbox_predictions_across_batches() -> None:
    metrics = TrainingMetrics()

    metrics.update_bbox(
        image_ids=["animal_001"],
        predictions=[
            DetectionPrediction(
                bbox=(10.0, 10.0, 50.0, 50.0),
                confidence=0.95,
            )
        ],
        targets=[
            (10.0, 10.0, 50.0, 50.0),
        ],
    )

    metrics.update_bbox(
        image_ids=["animal_002"],
        predictions=[
            DetectionPrediction(
                bbox=(20.0, 20.0, 40.0, 40.0),
                confidence=0.90,
            )
        ],
        targets=[
            (20.0, 20.0, 40.0, 40.0),
        ],
    )

    result = metrics.compute()

    assert result["map"] == pytest.approx(1.0)


def test_rejects_mismatched_bbox_prediction_lengths() -> None:
    metrics = TrainingMetrics()

    with pytest.raises(
        ValueError,
        match="image_ids and predictions must have equal length",
    ):
        metrics.update_bbox(
            image_ids=["animal_001"],
            predictions=[],
            targets=[(10.0, 10.0, 50.0, 50.0)],
        )


def test_rejects_mismatched_bbox_target_lengths() -> None:
    metrics = TrainingMetrics()

    with pytest.raises(
        ValueError,
        match="image_ids and targets must have equal length",
    ):
        metrics.update_bbox(
            image_ids=["animal_001"],
            predictions=[
                DetectionPrediction(
                    bbox=(10.0, 10.0, 50.0, 50.0),
                    confidence=0.95,
                )
            ],
            targets=[],
        )


def test_reset_clears_accumulated_metrics() -> None:
    metrics = TrainingMetrics()

    metrics.update_weight(
        predictions=[150.0],
        targets=[160.0],
    )

    metrics.reset()

    assert metrics.compute() == {}


def test_compute_can_return_weight_and_bbox_metrics() -> None:
    metrics = TrainingMetrics()

    metrics.update_weight(
        predictions=[150.0],
        targets=[160.0],
    )

    metrics.update_bbox(
        image_ids=["animal_001"],
        predictions=[
            DetectionPrediction(
                bbox=(10.0, 10.0, 50.0, 50.0),
                confidence=0.95,
            )
        ],
        targets=[
            (10.0, 10.0, 50.0, 50.0),
        ],
    )

    result = metrics.compute()

    assert result["mae"] == pytest.approx(10.0)
    assert result["map"] == pytest.approx(1.0)