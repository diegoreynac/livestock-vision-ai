from __future__ import annotations

import unittest

from src.evaluation.detection import (
    DetectionPrediction,
    calculate_average_precision,
    calculate_iou,
    calculate_map,
)


class TestDetectionMetrics(unittest.TestCase):
    def test_iou_identical_boxes(self) -> None:
        box = (1.0, 2.0, 3.0, 4.0)

        self.assertEqual(
            calculate_iou(box, box),
            1.0,
        )

    def test_iou_no_overlap(self) -> None:
        first = (0.0, 0.0, 1.0, 1.0)
        second = (2.0, 2.0, 1.0, 1.0)

        self.assertEqual(
            calculate_iou(first, second),
            0.0,
        )

    def test_iou_partial_overlap(self) -> None:
        first = (0.0, 0.0, 2.0, 2.0)
        second = (1.0, 1.0, 2.0, 2.0)

        self.assertAlmostEqual(
            calculate_iou(first, second),
            1.0 / 7.0,
        )

    def test_iou_zero_area_box(self) -> None:
        zero_area = (0.0, 0.0, 0.0, 2.0)
        normal = (0.0, 0.0, 2.0, 2.0)

        self.assertEqual(
            calculate_iou(zero_area, normal),
            0.0,
        )

    def test_iou_is_symmetric(self) -> None:
        first = (0.0, 0.0, 2.0, 2.0)
        second = (1.0, 1.0, 2.0, 2.0)

        self.assertEqual(
            calculate_iou(first, second),
            calculate_iou(second, first),
        )

    def test_average_precision_perfect_predictions(self) -> None:
        boxes = [
            (0.0, 0.0, 1.0, 1.0),
            (2.0, 2.0, 1.0, 1.0),
        ]

        predictions = [
            DetectionPrediction(boxes[0], 0.9),
            DetectionPrediction(boxes[1], 0.8),
        ]

        self.assertEqual(
            calculate_average_precision(
                predictions,
                boxes,
            ),
            1.0,
        )

    def test_average_precision_penalizes_high_confidence_false_positive(
        self,
    ) -> None:
        predictions = [
            DetectionPrediction(
                (3.0, 3.0, 1.0, 1.0),
                0.9,
            ),
            DetectionPrediction(
                (0.0, 0.0, 1.0, 1.0),
                0.8,
            ),
        ]

        targets = [
            (0.0, 0.0, 1.0, 1.0),
        ]

        self.assertAlmostEqual(
            calculate_average_precision(
                predictions,
                targets,
            ),
            0.5,
        )

    def test_average_precision_treats_duplicate_prediction_as_false_positive(
        self,
    ) -> None:
        first_box = (0.0, 0.0, 1.0, 1.0)
        second_box = (2.0, 2.0, 1.0, 1.0)

        predictions = [
            DetectionPrediction(
                first_box,
                0.9,
            ),
            DetectionPrediction(
                first_box,
                0.8,
            ),
            DetectionPrediction(
                second_box,
                0.7,
            ),
        ]

        self.assertAlmostEqual(
            calculate_average_precision(
                predictions,
                [
                    first_box,
                    second_box,
                ],
            ),
            5.0 / 6.0,
        )

    def test_map_keeps_image_identities_isolated(self) -> None:
        box = (0.0, 0.0, 1.0, 1.0)

        predictions = {
            "image-with-prediction": [
                DetectionPrediction(box, 0.9),
            ],
            "image-with-target": [],
        }

        targets = {
            "image-with-prediction": [],
            "image-with-target": [box],
        }

        self.assertEqual(
            calculate_map(
                predictions,
                targets,
            ),
            0.0,
        )

    def test_map_calculates_dataset_level_average_precision(
        self,
    ) -> None:
        first_box = (0.0, 0.0, 1.0, 1.0)
        second_box = (2.0, 2.0, 1.0, 1.0)

        predictions = {
            "first-image": [
                DetectionPrediction(
                    first_box,
                    0.9,
                ),
            ],
            "second-image": [
                DetectionPrediction(
                    second_box,
                    0.8,
                ),
            ],
        }

        targets = {
            "first-image": [first_box],
            "second-image": [second_box],
        }

        self.assertEqual(
            calculate_map(
                predictions,
                targets,
            ),
            1.0,
        )

    def test_map_supports_single_class_cattle_dataset(
        self,
    ) -> None:
        cattle_box = (
            0.0,
            0.0,
            1.0,
            1.0,
        )

        self.assertEqual(
            calculate_map(
                {
                    "cattle-image": [
                        DetectionPrediction(
                            cattle_box,
                            0.9,
                        ),
                    ],
                },
                {
                    "cattle-image": [
                        cattle_box,
                    ],
                },
            ),
            1.0,
        )

    def test_invalid_iou_threshold_raises_value_error(
        self,
    ) -> None:
        with self.assertRaises(ValueError):
            calculate_map(
                {},
                {},
                iou_threshold=1.1,
            )

    def test_average_precision_with_empty_predictions_is_zero(
        self,
    ) -> None:
        self.assertEqual(
            calculate_average_precision(
                [],
                [
                    (
                        0.0,
                        0.0,
                        1.0,
                        1.0,
                    ),
                ],
            ),
            0.0,
        )

    def test_average_precision_with_empty_ground_truth_is_zero(
        self,
    ) -> None:
        self.assertEqual(
            calculate_average_precision(
                [
                    DetectionPrediction(
                        (
                            0.0,
                            0.0,
                            1.0,
                            1.0,
                        ),
                        0.9,
                    ),
                ],
                [],
            ),
            0.0,
        )


if __name__ == "__main__":
    unittest.main()