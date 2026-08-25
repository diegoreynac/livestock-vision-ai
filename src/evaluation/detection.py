from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from dataclasses import dataclass

BoundingBox = tuple[float, float, float, float]


@dataclass(frozen=True)
class DetectionPrediction:
    """A single predicted detection."""

    bbox: BoundingBox
    confidence: float


def calculate_iou(
    predicted_bbox: BoundingBox, ground_truth_bbox: BoundingBox
) -> float:
    """Return IoU for bounding boxes expressed as ``(x, y, width, height)``."""
    predicted_area = _box_area(predicted_bbox)
    ground_truth_area = _box_area(ground_truth_bbox)

    if predicted_area == 0.0 or ground_truth_area == 0.0:
        return 0.0

    predicted_x_min, predicted_y_min, predicted_x_max, predicted_y_max = _corners(
        predicted_bbox
    )
    ground_truth_x_min, ground_truth_y_min, ground_truth_x_max, ground_truth_y_max = (
        _corners(ground_truth_bbox)
    )

    intersection_width = max(
        0.0,
        min(predicted_x_max, ground_truth_x_max)
        - max(predicted_x_min, ground_truth_x_min),
    )
    intersection_height = max(
        0.0,
        min(predicted_y_max, ground_truth_y_max)
        - max(predicted_y_min, ground_truth_y_min),
    )

    intersection_area = intersection_width * intersection_height
    union_area = predicted_area + ground_truth_area - intersection_area

    if union_area <= 0.0:
        return 0.0

    return min(1.0, max(0.0, intersection_area / union_area))


def calculate_average_precision(
    predictions: Sequence[DetectionPrediction],
    ground_truth_bboxes: Sequence[BoundingBox],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate AP for one image/sample at a confidence-ranked IoU threshold.

    This is a general, framework-independent object-detection metric
    implementation. It is not an exact reproduction of the official
    Ultralytics YOLO evaluator.
    """
    _validate_iou_threshold(iou_threshold)

    ranked_predictions = sorted(
        predictions,
        key=lambda prediction: prediction.confidence,
        reverse=True,
    )

    ranked_predictions_with_image = [
        (None, prediction)
        for prediction in ranked_predictions
    ]

    return _calculate_average_precision(
        ranked_predictions_with_image,
        {None: ground_truth_bboxes},
        iou_threshold,
    )


def calculate_map(
    predictions_by_image: Mapping[
        Hashable, Sequence[DetectionPrediction]
    ],
    ground_truths_by_image: Mapping[
        Hashable, Sequence[BoundingBox]
    ],
    iou_threshold: float = 0.5,
) -> float:
    """Calculate dataset-level mAP for a single detection class.

    Mapping keys identify images/samples. Predictions are matched only
    against ground-truth boxes carrying the same key, so detections
    cannot match boxes from another image.

    For the single-class livestock/cattle use case, mAP is the dataset
    AP for that class.

    This is a general, framework-independent object-detection metric
    implementation. It is not an exact reproduction of the official
    Ultralytics YOLO evaluator.
    """
    _validate_iou_threshold(iou_threshold)

    ranked_predictions = sorted(
        [
            (image_id, prediction)
            for image_id, predictions in predictions_by_image.items()
            for prediction in predictions
        ],
        key=lambda item: item[1].confidence,
        reverse=True,
    )

    return _calculate_average_precision(
        ranked_predictions,
        ground_truths_by_image,
        iou_threshold,
    )


def _calculate_average_precision(
    ranked_predictions: Sequence[
        tuple[Hashable | None, DetectionPrediction]
    ],
    ground_truths_by_image: Mapping[
        Hashable | None, Sequence[BoundingBox]
    ],
    iou_threshold: float,
) -> float:
    target_count = sum(
        len(boxes)
        for boxes in ground_truths_by_image.values()
    )

    if target_count == 0:
        return 0.0

    matched_ground_truths: dict[
        Hashable | None, set[int]
    ] = {}

    true_positives: list[int] = []
    false_positives: list[int] = []

    for image_id, prediction in ranked_predictions:
        ground_truth_bboxes = ground_truths_by_image.get(
            image_id,
            (),
        )

        matched_indexes = matched_ground_truths.setdefault(
            image_id,
            set(),
        )

        best_iou = 0.0
        best_index: int | None = None

        for index, ground_truth_bbox in enumerate(
            ground_truth_bboxes
        ):
            if index in matched_indexes:
                continue

            iou = calculate_iou(
                prediction.bbox,
                ground_truth_bbox,
            )

            if iou > best_iou:
                best_iou = iou
                best_index = index

        if (
            best_index is not None
            and best_iou >= iou_threshold
        ):
            matched_indexes.add(best_index)
            true_positives.append(1)
            false_positives.append(0)
        else:
            true_positives.append(0)
            false_positives.append(1)

    return _area_under_precision_recall_curve(
        true_positives,
        false_positives,
        target_count,
    )


def _corners(
    bbox: BoundingBox,
) -> tuple[float, float, float, float]:
    x, y, width, height = bbox

    return (
        x,
        y,
        x + width,
        y + height,
    )


def _box_area(
    bbox: BoundingBox,
) -> float:
    _, _, width, height = bbox

    return max(0.0, width) * max(
        0.0,
        height,
    )


def _validate_iou_threshold(
    iou_threshold: float,
) -> None:
    if not 0.0 <= iou_threshold <= 1.0:
        raise ValueError(
            "iou_threshold must be between 0.0 and 1.0"
        )


def _area_under_precision_recall_curve(
    true_positives: Sequence[int],
    false_positives: Sequence[int],
    target_count: int,
) -> float:
    cumulative_true_positives = 0
    cumulative_false_positives = 0

    recalls = [0.0]
    precisions = [1.0]

    for true_positive, false_positive in zip(
        true_positives,
        false_positives,
    ):
        cumulative_true_positives += true_positive
        cumulative_false_positives += false_positive

        recalls.append(
            cumulative_true_positives / target_count
        )

        precisions.append(
            cumulative_true_positives
            / (
                cumulative_true_positives
                + cumulative_false_positives
            )
        )

    for index in range(
        len(precisions) - 2,
        -1,
        -1,
    ):
        precisions[index] = max(
            precisions[index],
            precisions[index + 1],
        )

    return sum(
        (
            recalls[index]
            - recalls[index - 1]
        )
        * precisions[index]
        for index in range(
            1,
            len(recalls),
        )
    )