import os
import sys
import unittest

import numpy as np

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.augmentation import AugmentationConfig, eval_preprocess, training_preprocess, update_spatial_annotations


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        h, w = 32, 32
        base = np.linspace(0, 255, h * w, dtype=np.uint8).reshape((h, w))
        self.image = np.stack([base, base, base], axis=-1)

    def test_default_config_is_noop(self):
        cfg = AugmentationConfig()
        out = training_preprocess(self.image, cfg, seed=123)
        expected = self.image.astype(np.float32) / 255.0
        self.assertTrue(np.array_equal(out, expected), "Default config should not apply any augmentation")

    def test_reproducible_with_seed(self):
        cfg = AugmentationConfig(
            enable_flip=True,
            flip_prob=0.5,
            enable_brightness=True,
            brightness_range=(0.8, 1.2),
            enable_noise=True,
            noise_std=0.05,
        )
        out1 = training_preprocess(self.image, cfg, seed=123)
        out2 = training_preprocess(self.image, cfg, seed=123)
        self.assertTrue(np.array_equal(out1, out2), "Outputs differ with same seed")

    def test_conservative_reduces_magnitude(self):
        cfg_non = AugmentationConfig(
            enable_flip=True,
            flip_prob=0.5,
            enable_brightness=True,
            brightness_range=(0.8, 1.2),
            enable_noise=True,
            noise_std=0.08,
        )
        cfg_cons = AugmentationConfig(
            enable_flip=True,
            flip_prob=0.5,
            enable_brightness=True,
            brightness_range=(0.8, 1.2),
            enable_noise=True,
            noise_std=0.08,
            conservative=True,
        )
        out_non = training_preprocess(self.image, cfg_non, seed=99)
        out_cons = training_preprocess(self.image, cfg_cons, seed=99)
        img_f = self.image.astype(np.float32) / 255.0
        diff_non = np.mean(np.abs(out_non - img_f))
        diff_cons = np.mean(np.abs(out_cons - img_f))
        self.assertLess(diff_cons + 1e-8, diff_non, "Conservative should produce smaller average change")

    def test_eval_preprocess_deterministic_and_no_augmentation(self):
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        img[:, :3, :] = 50
        img[:, 3:, :] = 200
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0)
        out_eval = eval_preprocess(img, cfg)
        self.assertTrue(np.allclose(out_eval[:, :3, :], np.ones((4, 3, 3)) * (50 / 255.0)))

    def test_validation_invalid_params(self):
        cfg = AugmentationConfig()
        cfg.flip_prob = 1.5
        with self.assertRaises(ValueError):
            training_preprocess(self.image, cfg)

    def test_resize_changes_shape(self):
        cfg = AugmentationConfig(resize=(16, 8))
        out = eval_preprocess(self.image, cfg)
        self.assertEqual(out.shape[0], 16)
        self.assertEqual(out.shape[1], 8)

    def test_flip_is_reproducible(self):
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0, resize=(32, 32), enable_brightness=False, enable_noise=False)
        out = training_preprocess(self.image, cfg, seed=42)
        manual = (self.image.astype(np.float32) / 255.0)[:, ::-1, :]
        self.assertTrue(np.allclose(out, manual, atol=1e-6))

    def test_small_rotation_is_reproducible(self):
        cfg = AugmentationConfig(enable_rotation=True, rotation_range=(-15.0, 15.0), rotation_prob=1.0)
        out1 = training_preprocess(self.image, cfg, seed=7)
        out2 = training_preprocess(self.image, cfg, seed=7)
        self.assertTrue(np.array_equal(out1, out2), "Rotation should be reproducible with seed")

    def test_geometric_annotations_are_updated(self):
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0)
        annotations = {"boxes": np.array([[1, 2, 10, 12]], dtype=np.float32), "keypoints": np.array([[[5, 6], [9, 10]]], dtype=np.float32)}
        training_preprocess(self.image, cfg, seed=7, annotations=annotations)
        self.assertTrue(annotations["_spatial_annotations_updated"])
        self.assertTrue(np.any(annotations["boxes"] != np.array([[1, 2, 10, 12]], dtype=np.float32)))

    def test_photometric_transforms_do_not_update_annotations(self):
        cfg = AugmentationConfig(enable_brightness=True, brightness_range=(1.0, 1.0))
        annotations = {"boxes": np.array([[1, 2, 10, 12]], dtype=np.float32), "keypoints": np.array([[[5, 6], [9, 10]]], dtype=np.float32)}
        training_preprocess(self.image, cfg, seed=7, annotations=annotations)
        self.assertNotIn("_spatial_annotations_updated", annotations)
        self.assertTrue(np.array_equal(annotations["boxes"], np.array([[1, 2, 10, 12]], dtype=np.float32)))

    def test_resize_updates_bbox_and_keypoints(self):
        img = np.zeros((10, 20, 3), dtype=np.uint8)
        annotations = {
            "boxes": np.array([[2, 3, 8, 9]], dtype=np.float32),
            "keypoints": np.array([[[4, 5], [12, 7]]], dtype=np.float32),
        }
        eval_preprocess(img, AugmentationConfig(resize=(5, 10)), annotations=annotations)
        expected_boxes = np.array([[1.0, 1.5, 4.0, 4.5]], dtype=np.float32)
        expected_keypoints = np.array([[[2.0, 2.5], [6.0, 3.5]]], dtype=np.float32)
        self.assertTrue(np.allclose(annotations["boxes"], expected_boxes, atol=1e-6))
        self.assertTrue(np.allclose(annotations["keypoints"], expected_keypoints, atol=1e-6))

    def test_flip_and_rotation_compose_on_annotations(self):
        img = np.zeros((8, 10, 3), dtype=np.uint8)
        annotations = {
            "boxes": np.array([[2, 2, 6, 8]], dtype=np.float32),
            "keypoints": np.array([[[2, 2], [6, 8]]], dtype=np.float32),
        }
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0, enable_rotation=True, rotation_range=(10.0, 10.0), rotation_prob=1.0)
        training_preprocess(img, cfg, seed=2, annotations=annotations)

        manual = {
            "boxes": np.array([[2, 2, 6, 8]], dtype=np.float32),
            "keypoints": np.array([[[2, 2], [6, 8]]], dtype=np.float32),
        }
        update_spatial_annotations(manual, transform_name="horizontal_flip", source_shape=(8, 10), target_shape=(8, 10))
        update_spatial_annotations(manual, transform_name="small_rotation", source_shape=(8, 10), target_shape=(8, 10), angle_deg=10.0)

        self.assertEqual([step["transform"] for step in annotations["_geometric_transform_history"]], ["horizontal_flip", "small_rotation"])
        self.assertTrue(np.allclose(annotations["boxes"], manual["boxes"], atol=1e-6))
        self.assertTrue(np.allclose(annotations["keypoints"], manual["keypoints"], atol=1e-6))

    def test_multiple_geometric_transforms_are_applied_in_order(self):
        img = np.zeros((12, 16, 3), dtype=np.uint8)
        annotations = {
            "boxes": np.array([[2, 3, 8, 9]], dtype=np.float32),
            "keypoints": np.array([[[4, 5], [10, 11]]], dtype=np.float32),
        }
        cfg = AugmentationConfig(
            resize=(8, 12),
            enable_flip=True,
            flip_prob=1.0,
            enable_rotation=True,
            rotation_range=(15.0, 15.0),
            rotation_prob=1.0,
            enable_zoom=True,
            zoom_range=(1.0, 1.0),
            zoom_prob=1.0,
        )
        training_preprocess(img, cfg, seed=11, annotations=annotations)
        self.assertEqual(
            [step["transform"] for step in annotations["_geometric_transform_history"]],
            ["resize", "horizontal_flip", "small_rotation", "zoom_scale"],
        )

    def test_photometric_transforms_preserve_annotations(self):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        annotations = {
            "boxes": np.array([[1, 2, 6, 7]], dtype=np.float32),
            "keypoints": np.array([[[3, 4], [6, 7]]], dtype=np.float32),
        }
        cfg = AugmentationConfig(enable_brightness=True, brightness_range=(1.0, 1.0), enable_noise=True, noise_std=0.0)
        training_preprocess(img, cfg, seed=5, annotations=annotations)
        self.assertNotIn("_spatial_annotations_updated", annotations)
        self.assertTrue(np.array_equal(annotations["boxes"], np.array([[1, 2, 6, 7]], dtype=np.float32)))
        self.assertTrue(np.array_equal(annotations["keypoints"], np.array([[[3, 4], [6, 7]]], dtype=np.float32)))

    def test_keypoint_transform_tracks_coordinates(self):
        img = np.zeros((6, 10, 3), dtype=np.uint8)
        annotations = {"keypoints": np.array([[[1, 2], [5, 4]]], dtype=np.float32)}
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0)
        training_preprocess(img, cfg, seed=4, annotations=annotations)
        expected = np.array([[[9, 2], [5, 4]]], dtype=np.float32)
        self.assertTrue(np.allclose(annotations["keypoints"], expected, atol=1e-6))

    def test_seeded_preprocess_is_repeatable_across_multiple_geometry_ops(self):
        img = np.zeros((16, 20, 3), dtype=np.uint8)
        cfg = AugmentationConfig(
            resize=(12, 16),
            enable_flip=True,
            flip_prob=0.8,
            enable_rotation=True,
            rotation_range=(-12.0, 12.0),
            rotation_prob=0.8,
            enable_zoom=True,
            zoom_range=(0.95, 1.05),
            zoom_prob=0.8,
            enable_brightness=True,
            brightness_range=(0.9, 1.1),
            enable_noise=True,
            noise_std=0.02,
        )
        out1 = training_preprocess(img, cfg, seed=123)
        out2 = training_preprocess(img, cfg, seed=123)
        self.assertTrue(np.array_equal(out1, out2))

    def test_normalization(self):
        img = np.ones((4, 4, 3), dtype=np.uint8) * 128
        cfg = AugmentationConfig(normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.25, 0.25, 0.25))
        out = eval_preprocess(img, cfg)
        expected_value = (128 / 255.0 - 0.5) / 0.25
        self.assertTrue(np.allclose(out, expected_value, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
