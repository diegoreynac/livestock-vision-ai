import os
import sys
import unittest

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.bbox_preprocessing import (
    BoundingBox,
    clip_bbox,
    coco_to_xyxy,
    normalize_bbox,
    validate_bbox,
    validate_image_dimensions,
)


class TestBBoxPreprocessing(unittest.TestCase):
    def test_coco_to_xyxy_conversion(self):
        bbox = coco_to_xyxy([20, 30, 40, 50], image_width=1000, image_height=800)
        self.assertEqual(bbox.xyxy, [20.0, 30.0, 60.0, 80.0])
        self.assertEqual(bbox.image_width, 1000)
        self.assertEqual(bbox.image_height, 800)

    def test_valid_bbox_acceptance(self):
        validated = validate_bbox(10.0, 20.0, 30.0, 45.0)
        self.assertEqual(validated, (10.0, 20.0, 30.0, 45.0))
        self.assertEqual(BoundingBox(10.0, 20.0, 30.0, 45.0).xyxy, [10.0, 20.0, 30.0, 45.0])

    def test_invalid_bbox_rejection(self):
        with self.assertRaises(ValueError):
            validate_bbox(30.0, 20.0, 10.0, 45.0)
        with self.assertRaises(ValueError):
            validate_bbox(10.0, 20.0, 10.0, 45.0)
        with self.assertRaises(ValueError):
            validate_bbox(10.0, float("nan"), 20.0, 30.0)

    def test_negative_coordinates(self):
        with self.assertRaises(ValueError):
            validate_bbox(-1.0, 10.0, 20.0, 30.0)
        with self.assertRaises(ValueError):
            coco_to_xyxy([-1, 10, 30, 40], image_width=100, image_height=100)

    def test_zero_or_negative_width_or_height(self):
        for bad_bbox in ([10, 10, 0, 20], [10, 10, 20, 0], [10, 10, -5, 20], [10, 10, 20, -5]):
            with self.assertRaises(ValueError):
                coco_to_xyxy(bad_bbox, image_width=100, image_height=100)

    def test_clipping_to_image_boundaries(self):
        raw = [10, 15, 130, 200]
        clipped = clip_bbox(raw, image_width=100, image_height=120)
        self.assertEqual(clipped.xyxy, [10.0, 15.0, 100.0, 120.0])

    def test_clipping_rejects_negative_xyxy_coordinates(self):
        with self.assertRaises(ValueError):
            clip_bbox([-10, 15, 130, 200], image_width=100, image_height=120)

    def test_clipping_rejects_non_finite_xyxy_coordinates(self):
        with self.assertRaises(ValueError):
            clip_bbox([10, 15, float("inf"), 200], image_width=100, image_height=120)

    def test_clipping_does_not_mutate_raw_sequence(self):
        raw = [10, 15, 130, 200]
        clip_bbox(raw, image_width=100, image_height=120)
        self.assertEqual(raw, [10, 15, 130, 200])

    def test_clipping_does_not_mutate_bounding_box(self):
        box = BoundingBox(10, 15, 130, 200)
        clipped = clip_bbox(box, image_width=100, image_height=120)
        self.assertEqual(box.xyxy, [10.0, 15.0, 130.0, 200.0])
        self.assertEqual(clipped.xyxy, [10.0, 15.0, 100.0, 120.0])

    def test_normalization_to_unit_interval(self):
        box = BoundingBox(50, 20, 100, 80, image_width=200, image_height=100)
        self.assertEqual(normalize_bbox(box), [0.25, 0.2, 0.5, 0.8])
        self.assertEqual(normalize_bbox([50, 20, 100, 80], image_width=200, image_height=100), [0.25, 0.2, 0.5, 0.8])

    def test_normalization_rejects_out_of_bounds_box(self):
        box = BoundingBox(10, 15, 130, 200)
        with self.assertRaises(ValueError):
            normalize_bbox(box, image_width=100, image_height=120)

    def test_bbox_at_image_boundary(self):
        box = BoundingBox(0, 0, 640, 480, image_width=640, image_height=480)
        self.assertEqual(box.xyxy, [0.0, 0.0, 640.0, 480.0])
        self.assertEqual(normalize_bbox(box), [0.0, 0.0, 1.0, 1.0])

    def test_invalid_image_dimensions(self):
        with self.assertRaises(ValueError):
            validate_image_dimensions(0, 100)
        with self.assertRaises(ValueError):
            validate_image_dimensions(100, 0)
        with self.assertRaises(ValueError):
            validate_image_dimensions(-10, 100)
        with self.assertRaises(ValueError):
            validate_image_dimensions(10.5, 100)

    def test_deterministic_behavior(self):
        first = coco_to_xyxy([10, 20, 30, 40], image_width=640, image_height=480)
        second = coco_to_xyxy([10, 20, 30, 40], image_width=640, image_height=480)
        self.assertEqual(first, second)
        self.assertEqual(first.xyxy, second.xyxy)

    def test_coco_conversion_rejects_out_of_bounds_without_mutating_input(self):
        raw = [90, 70, 20, 40]
        with self.assertRaises(ValueError):
            coco_to_xyxy(raw, image_width=100, image_height=100)
        self.assertEqual(raw, [90, 70, 20, 40])


if __name__ == "__main__":
    unittest.main()
