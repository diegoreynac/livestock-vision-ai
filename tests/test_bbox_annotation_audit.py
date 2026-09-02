import json
import tempfile
import unittest
from pathlib import Path

from src.analysis.bbox_annotation_audit import (
    BBoxAnnotationAudit,
    CandidateBoundingBoxXYXY,
    ImageAnnotationClass,
    MIN_KEYPOINTS_FOR_BBOX,
    derive_candidate_bbox_from_keypoints,
)
from src.evaluation.detection import calculate_iou


def _coco(images, annotations, categories=None):
    return {
        "images": images,
        "annotations": annotations,
        "categories": categories or [{"id": 1, "name": "cow"}],
    }


class TestBBoxAnnotationAudit(unittest.TestCase):

    def setUp(self) -> None:
        self.audit = BBoxAnnotationAudit()

    # --------------------------------------------------
    # Image with bbox
    # --------------------------------------------------

    def test_image_with_valid_bbox(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 10,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [1.0, 2.0, 10.0, 20.0],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(1)

        self.assertIsNotNone(image)
        self.assertEqual(image.classification, ImageAnnotationClass.BBOX_AVAILABLE)
        self.assertTrue(image.has_valid_bbox)
        self.assertEqual(report.statistics.annotations_with_bbox, 1)
        self.assertEqual(report.statistics.annotations_without_bbox, 0)

    # --------------------------------------------------
    # Image without bbox but with keypoints
    # --------------------------------------------------

    def test_image_without_bbox_with_sufficient_keypoints(self) -> None:
        data = _coco(
            images=[{"id": 2, "file_name": "b.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 11,
                    "image_id": 2,
                    "category_id": 1,
                    "keypoints": [
                        10.0, 20.0, 2,
                        30.0, 40.0, 2,
                        50.0, 60.0, 1,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(2)

        self.assertIsNotNone(image)
        self.assertFalse(image.has_valid_bbox)
        self.assertTrue(image.has_keypoints)
        self.assertTrue(image.keypoints_sufficient_for_bbox)
        self.assertEqual(
            image.classification,
            ImageAnnotationClass.BBOX_MISSING_KEYPOINTS_AVAILABLE,
        )
        self.assertEqual(
            report.statistics.annotations_missing_bbox_with_keypoints, 1
        )
        self.assertEqual(
            report.statistics.annotations_keypoints_sufficient_for_bbox, 1
        )

    def test_keypoints_below_minimum_are_not_sufficient(self) -> None:
        self.assertGreaterEqual(MIN_KEYPOINTS_FOR_BBOX, 2)

        data = _coco(
            images=[{"id": 3, "file_name": "c.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 12,
                    "image_id": 3,
                    "category_id": 1,
                    # Only one labeled/finite keypoint; the rest are
                    # not-labeled (visibility 0), so they cannot count.
                    "keypoints": [
                        10.0, 20.0, 2,
                        0.0, 0.0, 0,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(3)

        self.assertFalse(image.keypoints_sufficient_for_bbox)
        self.assertEqual(
            image.classification,
            ImageAnnotationClass.BBOX_MISSING_KEYPOINTS_MISSING,
        )

    # --------------------------------------------------
    # Image without bbox and without keypoints
    # --------------------------------------------------

    def test_image_without_bbox_without_keypoints(self) -> None:
        data = _coco(
            images=[{"id": 4, "file_name": "d.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 13,
                    "image_id": 4,
                    "category_id": 1,
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(4)

        self.assertFalse(image.has_valid_bbox)
        self.assertFalse(image.has_keypoints)
        self.assertEqual(
            image.classification,
            ImageAnnotationClass.BBOX_MISSING_KEYPOINTS_MISSING,
        )

    # --------------------------------------------------
    # Image without annotations at all
    # --------------------------------------------------

    def test_image_without_annotations(self) -> None:
        data = _coco(
            images=[{"id": 5, "file_name": "e.jpg", "width": 100, "height": 200}],
            annotations=[],
        )

        report = self.audit.audit(data)
        image = report.get_image(5)

        self.assertIsNotNone(image)
        self.assertFalse(image.has_annotations)
        self.assertEqual(
            image.classification, ImageAnnotationClass.ANNOTATION_MISSING
        )
        self.assertEqual(report.statistics.images_without_annotations, 1)
        self.assertEqual(report.statistics.images_with_annotations, 0)

    # --------------------------------------------------
    # Invalid / malformed bbox
    # --------------------------------------------------

    def test_invalid_bbox_negative_size(self) -> None:
        data = _coco(
            images=[{"id": 6, "file_name": "f.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 14,
                    "image_id": 6,
                    "category_id": 1,
                    "bbox": [1.0, 2.0, -5.0, 20.0],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(6)

        self.assertFalse(image.has_valid_bbox)
        self.assertEqual(image.classification, ImageAnnotationClass.BBOX_INVALID)
        self.assertEqual(report.statistics.annotations_with_invalid_bbox, 1)

    def test_invalid_bbox_wrong_length(self) -> None:
        data = _coco(
            images=[{"id": 7, "file_name": "g.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 15,
                    "image_id": 7,
                    "category_id": 1,
                    "bbox": [1.0, 2.0, 3.0],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(7)

        self.assertFalse(image.has_valid_bbox)
        self.assertEqual(image.classification, ImageAnnotationClass.BBOX_INVALID)

    # --------------------------------------------------
    # Keypoints with visibility information
    # --------------------------------------------------

    def test_keypoint_visibility_flags_are_reported(self) -> None:
        data = _coco(
            images=[{"id": 8, "file_name": "h.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 16,
                    "image_id": 8,
                    "category_id": 1,
                    "keypoints": [
                        1.0, 2.0, 0,   # not labeled
                        3.0, 4.0, 1,   # labeled, occluded
                        5.0, 6.0, 2,   # labeled, visible
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(8)
        annotation = image.annotations[0]

        self.assertEqual(annotation.keypoint_count, 3)
        self.assertFalse(annotation.keypoints[0].is_labeled)
        self.assertTrue(annotation.keypoints[1].is_labeled)
        self.assertTrue(annotation.keypoints[2].is_labeled)
        self.assertEqual(annotation.valid_keypoint_count, 2)

    # --------------------------------------------------
    # Multiple annotations for one image
    # --------------------------------------------------

    def test_multiple_annotations_for_one_image(self) -> None:
        data = _coco(
            images=[{"id": 9, "file_name": "i.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 17,
                    "image_id": 9,
                    "category_id": 1,
                },
                {
                    "id": 18,
                    "image_id": 9,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                },
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(9)

        self.assertEqual(image.annotation_count, 2)
        self.assertEqual(image.annotation_ids, [17, 18])
        # At least one annotation has a valid bbox -> image classified
        # as bbox available even though another annotation lacks one.
        self.assertEqual(image.classification, ImageAnnotationClass.BBOX_AVAILABLE)
        self.assertEqual(report.statistics.total_annotations, 2)

    # --------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------

    def test_aggregate_statistics(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
                {"id": 3, "file_name": "c.jpg", "width": 10, "height": 10},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "keypoints": [1.0, 1.0, 2, 2.0, 2.0, 2],
                },
                # image 3 has no annotation entry at all
            ],
        )

        report = self.audit.audit(data)
        stats = report.statistics

        self.assertEqual(stats.total_images, 3)
        self.assertEqual(stats.images_with_annotations, 2)
        self.assertEqual(stats.images_without_annotations, 1)
        self.assertEqual(stats.total_annotations, 2)
        self.assertEqual(stats.annotations_with_bbox, 1)
        self.assertEqual(stats.annotations_without_bbox, 1)
        self.assertEqual(stats.annotations_with_keypoints, 1)
        self.assertEqual(stats.annotations_without_keypoints, 1)
        self.assertEqual(stats.annotations_missing_bbox_with_keypoints, 1)
        self.assertEqual(stats.annotations_keypoints_sufficient_for_bbox, 1)

    # --------------------------------------------------
    # Image-level lookup
    # --------------------------------------------------

    def test_get_image_returns_none_for_unknown_id(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[],
        )

        report = self.audit.audit(data)

        self.assertIsNone(report.get_image(999))
        self.assertIsNotNone(report.get_image(1))

    def test_images_by_classification(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                },
            ],
        )

        report = self.audit.audit(data)

        bbox_available = report.images_by_classification(
            ImageAnnotationClass.BBOX_AVAILABLE
        )
        annotation_missing = report.images_by_classification(
            ImageAnnotationClass.ANNOTATION_MISSING
        )

        self.assertEqual([item.image_id for item in bbox_available], [1])
        self.assertEqual([item.image_id for item in annotation_missing], [2])

    def test_category_name_lookup(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 7,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                }
            ],
            categories=[{"id": 7, "name": "bull"}],
        )

        report = self.audit.audit(data)
        image = report.get_image(1)

        self.assertEqual(image.category_ids, [7])
        self.assertEqual(image.category_names, ["bull"])

    def test_to_dict_is_serializable_shape(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "segmentation": [[1.0, 2.0, 3.0, 4.0]],
                }
            ],
        )

        report = self.audit.audit(data)
        result = report.to_dict()

        self.assertIn("statistics", result)
        self.assertIn("images", result)
        self.assertEqual(len(result["images"]), 1)
        annotation_dict = result["images"][0]["annotations"][0]
        self.assertTrue(annotation_dict["has_segmentation"])
        self.assertEqual(annotation_dict["segmentation_type"], "polygon")


class TestCandidateBBoxDerivation(unittest.TestCase):
    """
    Unit tests for the TEMPORARY candidate BBox derivation and its
    comparison (via IoU) against ground-truth BBoxes. Nothing here
    reconstructs or writes a BBox back into any annotation.
    """

    def setUp(self) -> None:
        self.audit = BBoxAnnotationAudit()

    # --------------------------------------------------
    # Candidate bbox from valid keypoints
    # --------------------------------------------------

    def test_candidate_bbox_from_valid_keypoints(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [
                        10.0, 20.0, 2,
                        30.0, 5.0, 2,
                        15.0, 40.0, 1,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        self.assertIsNotNone(annotation.candidate_bbox)
        candidate = annotation.candidate_bbox
        self.assertEqual(candidate.x_min, 10.0)
        self.assertEqual(candidate.x_max, 30.0)
        self.assertEqual(candidate.y_min, 5.0)
        self.assertEqual(candidate.y_max, 40.0)
        self.assertEqual(candidate.source_keypoint_count, 3)

    # --------------------------------------------------
    # Insufficient keypoints
    # --------------------------------------------------

    def test_insufficient_keypoints_yields_no_candidate(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [10.0, 20.0, 2],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        self.assertIsNone(annotation.candidate_bbox)

    def test_derive_candidate_bbox_helper_respects_min_keypoints(self) -> None:
        self.assertGreaterEqual(MIN_KEYPOINTS_FOR_BBOX, 2)

    # --------------------------------------------------
    # Invisible / not-labeled keypoints are excluded
    # --------------------------------------------------

    def test_not_labeled_keypoints_are_excluded_from_candidate(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [
                        10.0, 20.0, 2,   # labeled, visible
                        999.0, 999.0, 0,  # not labeled -> must be ignored
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        # Only one usable (labeled) keypoint remains -> below the
        # minimum required to span a box.
        self.assertIsNone(annotation.candidate_bbox)

    def test_occluded_but_labeled_keypoints_are_used(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [
                        10.0, 20.0, 1,  # labeled, occluded
                        30.0, 40.0, 2,  # labeled, visible
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        self.assertIsNotNone(annotation.candidate_bbox)
        self.assertEqual(annotation.candidate_bbox.source_keypoint_count, 2)

    # --------------------------------------------------
    # Non-finite coordinates
    # --------------------------------------------------

    def test_non_finite_coordinates_are_excluded(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "keypoints": [
                        10.0, 20.0, 2,
                        float("nan"), 40.0, 2,
                        float("inf"), 40.0, 2,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        # Only one finite, labeled keypoint remains -> insufficient.
        self.assertIsNone(annotation.candidate_bbox)

    def test_derive_candidate_bbox_from_keypoints_direct_call(self) -> None:
        from src.analysis.bbox_annotation_audit import KeypointAuditEntry

        keypoints = [
            KeypointAuditEntry(0, 1.0, 2.0, 2, True, True),
            KeypointAuditEntry(1, float("nan"), 5.0, 2, False, True),
            KeypointAuditEntry(2, 9.0, 9.0, 2, True, True),
        ]

        candidate = derive_candidate_bbox_from_keypoints(keypoints)

        self.assertIsInstance(candidate, CandidateBoundingBoxXYXY)
        self.assertEqual(candidate.source_keypoint_count, 2)
        self.assertEqual(candidate.x_min, 1.0)
        self.assertEqual(candidate.x_max, 9.0)

    # --------------------------------------------------
    # Candidate bbox stays separate from the original annotation
    # --------------------------------------------------

    def test_candidate_bbox_does_not_mutate_original_bbox(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [
                        10.0, 20.0, 2,
                        30.0, 40.0, 2,
                    ],
                }
            ],
        )

        original_bbox_list = data["annotations"][0]["bbox"]

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        # The original raw payload must remain untouched.
        self.assertEqual(data["annotations"][0]["bbox"], [0.0, 0.0, 0.0, 0.0])
        self.assertIs(data["annotations"][0]["bbox"], original_bbox_list)

        # The audit still reports the original bbox as invalid...
        self.assertFalse(annotation.bbox_is_valid)
        self.assertEqual(annotation.bbox_values, (0.0, 0.0, 0.0, 0.0))

        # ...while the candidate is a distinct, independent object.
        self.assertIsNotNone(annotation.candidate_bbox)
        self.assertIsInstance(annotation.candidate_bbox, CandidateBoundingBoxXYXY)
        self.assertNotEqual(
            annotation.candidate_bbox.as_xywh, annotation.bbox_values
        )

    def test_candidate_bbox_is_immutable(self) -> None:
        candidate = CandidateBoundingBoxXYXY(0.0, 0.0, 10.0, 10.0, 2)

        with self.assertRaises(Exception):
            candidate.x_min = 5.0  # type: ignore[misc]

    # --------------------------------------------------
    # IoU calculation
    # --------------------------------------------------

    def test_iou_perfect_match(self) -> None:
        iou = calculate_iou((0.0, 0.0, 10.0, 10.0), (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(iou, 1.0)

    def test_iou_no_overlap(self) -> None:
        iou = calculate_iou((0.0, 0.0, 5.0, 5.0), (100.0, 100.0, 5.0, 5.0))
        self.assertEqual(iou, 0.0)

    # --------------------------------------------------
    # Candidate bbox against a known ground truth bbox
    # --------------------------------------------------

    def test_iou_reported_when_ground_truth_and_keypoints_both_exist(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10.0, 10.0, 20.0, 20.0],
                    "keypoints": [
                        10.0, 10.0, 2,
                        30.0, 30.0, 2,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        self.assertIsNotNone(annotation.candidate_bbox)
        self.assertIsNotNone(annotation.iou_with_ground_truth)
        # Candidate == ground truth here (both are [10,10,20,20] xywh).
        self.assertAlmostEqual(annotation.iou_with_ground_truth, 1.0)

    def test_iou_not_reported_when_ground_truth_bbox_is_invalid(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [
                        10.0, 10.0, 2,
                        30.0, 30.0, 2,
                    ],
                }
            ],
        )

        report = self.audit.audit(data)
        annotation = report.get_annotation(1, 1)

        self.assertIsNotNone(annotation.candidate_bbox)
        self.assertIsNone(annotation.iou_with_ground_truth)

    # --------------------------------------------------
    # Evidence report: invalid bbox + usable keypoints
    # --------------------------------------------------

    def test_annotations_with_candidate_from_invalid_bbox_report(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [1.0, 1.0, 2, 5.0, 5.0, 2],
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "keypoints": [1.0, 1.0, 2, 5.0, 5.0, 2],
                },
            ],
        )

        report = self.audit.audit(data)
        pairs = report.annotations_with_candidate_from_invalid_bbox()

        self.assertEqual(len(pairs), 1)
        image, annotation = pairs[0]
        self.assertEqual(image.image_id, 1)
        self.assertEqual(annotation.annotation_id, 1)

    # --------------------------------------------------
    # Aggregate statistics
    # --------------------------------------------------

    def test_aggregate_statistics_for_candidates_and_iou(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
                {"id": 3, "file_name": "c.jpg", "width": 10, "height": 10},
            ],
            annotations=[
                # Valid bbox + valid keypoints -> IoU sample.
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 10.0, 10.0, 2],
                },
                # Invalid bbox + usable keypoints -> candidate only.
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [1.0, 1.0, 2, 9.0, 9.0, 2],
                },
                # Invalid bbox + insufficient keypoints -> no candidate.
                {
                    "id": 3,
                    "image_id": 3,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [1.0, 1.0, 2],
                },
            ],
        )

        report = self.audit.audit(data)
        stats = report.statistics

        self.assertEqual(stats.annotations_with_valid_bbox_and_keypoints, 1)
        self.assertEqual(
            stats.annotations_with_invalid_bbox_and_usable_keypoints, 1
        )
        self.assertEqual(stats.candidate_bboxes_derived, 2)
        self.assertEqual(stats.iou_sample_count, 1)
        self.assertEqual(stats.iou_min, 1.0)
        self.assertEqual(stats.iou_max, 1.0)
        self.assertEqual(stats.iou_mean, 1.0)
        self.assertEqual(stats.iou_median, 1.0)
        self.assertEqual(stats.iou_stdev, 0.0)
        self.assertEqual(stats.iou_at_or_above_0_5, 1)
        self.assertEqual(stats.iou_below_0_5, 0)

    def test_iou_statistics_with_multiple_samples(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 10.0, 10.0, 2],
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 4.0, 4.0, 2],
                },
            ],
        )

        report = self.audit.audit(data)
        stats = report.statistics

        self.assertEqual(stats.iou_sample_count, 2)
        self.assertIsNotNone(stats.iou_q1)
        self.assertIsNotNone(stats.iou_q3)
        self.assertLessEqual(stats.iou_min, stats.iou_max)

    # --------------------------------------------------
    # Multiple annotations per image
    # --------------------------------------------------

    def test_multiple_annotations_per_image_each_get_own_candidate(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [1.0, 1.0, 2, 5.0, 5.0, 2],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 8.0, 8.0],
                    "keypoints": [0.0, 0.0, 2, 8.0, 8.0, 2],
                },
            ],
        )

        report = self.audit.audit(data)
        image = report.get_image(1)

        self.assertEqual(image.annotation_count, 2)
        first = image.get_annotation(1)
        second = image.get_annotation(2)

        self.assertIsNotNone(first.candidate_bbox)
        self.assertIsNone(first.iou_with_ground_truth)

        self.assertIsNotNone(second.candidate_bbox)
        self.assertIsNotNone(second.iou_with_ground_truth)
        self.assertAlmostEqual(second.iou_with_ground_truth, 1.0)

    # --------------------------------------------------
    # Annotation-level lookup (inspect_annotation)
    # --------------------------------------------------

    def test_inspect_annotation_returns_full_context(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 100, "height": 200}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 10.0, 10.0, 2],
                }
            ],
        )

        report = self.audit.audit(data)
        inspection = report.inspect_annotation(1, 1)

        self.assertEqual(inspection["image_id"], 1)
        self.assertEqual(inspection["filename"], "a.jpg")
        self.assertEqual(inspection["width"], 100)
        self.assertEqual(inspection["height"], 200)
        self.assertEqual(inspection["annotation_id"], 1)
        self.assertEqual(inspection["bbox_values"], (0.0, 0.0, 10.0, 10.0))
        self.assertIsNotNone(inspection["candidate_bbox"])
        self.assertIsNotNone(inspection["iou_with_ground_truth"])

    def test_inspect_annotation_returns_none_for_unknown_pair(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "a.jpg", "width": 10, "height": 10}],
            annotations=[],
        )

        report = self.audit.audit(data)

        self.assertIsNone(report.inspect_annotation(1, 999))
        self.assertIsNone(report.inspect_annotation(999, 1))

    # --------------------------------------------------
    # Cross-file counting and IoU range reconciliation
    # --------------------------------------------------

    def test_file_aggregate_keeps_image_ids_source_scoped(self) -> None:
        first = _coco(
            images=[{"id": 1, "file_name": "first.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 10.0, 10.0, 2],
                },
                {
                    "id": 2,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 0.0, 0.0],
                    "keypoints": [1.0, 1.0, 2, 5.0, 5.0, 2],
                },
            ],
        )
        second = _coco(
            images=[{"id": 1, "file_name": "second.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 3,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                    "keypoints": [0.0, 0.0, 2, 5.0, 5.0, 2],
                },
            ],
        )

        with tempfile.TemporaryDirectory() as directory:
            first_path = Path(directory) / "first.json"
            second_path = Path(directory) / "second.json"
            first_path.write_text(json.dumps(first), encoding="utf-8")
            second_path.write_text(json.dumps(second), encoding="utf-8")

            collection = self.audit.audit_files([first_path, second_path])

        stats = collection.statistics
        self.assertEqual(stats.total_images, 2)
        self.assertEqual(stats.total_unique_images, 2)
        self.assertEqual(stats.unique_image_id_values, 1)
        self.assertEqual(stats.total_annotations, 3)
        self.assertEqual(stats.indexed_annotations, 3)
        self.assertEqual(stats.orphan_annotations, 0)
        self.assertEqual(stats.annotations_per_image, {2: 1, 1: 1})
        self.assertEqual(stats.images_with_multiple_annotations, 1)
        self.assertEqual(stats.annotations_with_bbox, 2)
        self.assertEqual(stats.annotations_with_invalid_bbox, 1)
        self.assertEqual(stats.candidate_bboxes_derived, 3)
        self.assertEqual(
            collection.inspect_annotation(str(first_path), 1, 2)["filename"],
            "first.jpg",
        )

    def test_raw_annotations_are_reconciled_with_orphan_annotations(self) -> None:
        data = _coco(
            images=[{"id": 1, "file_name": "indexed.jpg", "width": 10, "height": 10}],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                },
                {
                    "id": 2,
                    "image_id": 999,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 5.0, 5.0],
                },
            ],
        )

        report = self.audit.audit(data, source_file="sample.json")
        stats = report.statistics

        self.assertEqual(stats.total_images, 1)
        self.assertEqual(stats.total_unique_images, 1)
        self.assertEqual(stats.total_annotations, 2)
        self.assertEqual(stats.indexed_annotations, 1)
        self.assertEqual(stats.orphan_annotations, 1)
        self.assertEqual(stats.annotations_with_bbox, 1)
        self.assertEqual(stats.raw_annotations_with_valid_bbox, 2)
        self.assertEqual(stats.raw_annotations_with_invalid_bbox, 0)

    def test_iou_case_reports_apply_descriptive_ranges(self) -> None:
        data = _coco(
            images=[
                {"id": 1, "file_name": "low.jpg", "width": 100, "height": 100},
                {"id": 2, "file_name": "middle.jpg", "width": 100, "height": 100},
                {"id": 3, "file_name": "high.jpg", "width": 100, "height": 100},
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [20.0, 20.0, 2, 25.0, 25.0, 2],
                },
                {
                    "id": 2,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 8.0, 8.0, 2],
                },
                {
                    "id": 3,
                    "image_id": 3,
                    "category_id": 1,
                    "bbox": [0.0, 0.0, 10.0, 10.0],
                    "keypoints": [0.0, 0.0, 2, 10.0, 10.0, 2],
                },
            ],
        )

        report = self.audit.audit(data, source_file="sample.json")
        from src.analysis.bbox_annotation_audit import BBoxAnnotationAuditCollectionReport

        collection = BBoxAnnotationAuditCollectionReport(
            files=[report],
            statistics=report.statistics,
        )

        below = collection.iou_cases_below(0.50)
        middle = collection.iou_cases_in_range(0.50, 0.75)

        self.assertEqual([case.annotation_id for case in below], [1])
        self.assertEqual([case.annotation_id for case in middle], [2])
        self.assertEqual(below[0].source_file, "sample.json")
        self.assertEqual(below[0].filename, "low.jpg")
        self.assertEqual(below[0].ground_truth_bbox, (0.0, 0.0, 10.0, 10.0))
        self.assertEqual(below[0].candidate_bbox.as_xywh, (20.0, 20.0, 5.0, 5.0))
        self.assertEqual(below[0].usable_keypoint_count, 2)
        self.assertLess(below[0].iou, 0.50)
        self.assertGreaterEqual(middle[0].iou, 0.50)
        self.assertLess(middle[0].iou, 0.75)


if __name__ == "__main__":
    unittest.main()
