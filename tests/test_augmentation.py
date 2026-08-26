import os
import sys
import unittest
import numpy as np

# Ensure src is on path so tests can import training.augmentation
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SRC_PATH = os.path.join(REPO_ROOT, "src")
if SRC_PATH not in sys.path:
    sys.path.insert(0, SRC_PATH)

from training.augmentation import (
    training_preprocess,
    eval_preprocess,
    AugmentationConfig,
)


class TestAugmentation(unittest.TestCase):
    def setUp(self):
        # deterministic test image: a gradient with three channels
        h, w = 32, 32
        base = np.linspace(0, 255, h * w, dtype=np.uint8).reshape((h, w))
        self.image = np.stack([base, base, base], axis=-1)

    def test_reproducible_with_seed(self):
        cfg = AugmentationConfig(enable_flip=True, flip_prob=0.5, enable_brightness=True, brightness_range=(0.8, 1.2), enable_noise=True, noise_std=0.05)
        out1 = training_preprocess(self.image, cfg, seed=123)
        out2 = training_preprocess(self.image, cfg, seed=123)
        self.assertTrue(np.array_equal(out1, out2), "Outputs differ with same seed")

    def test_conservative_reduces_magnitude(self):
        cfg_non = AugmentationConfig(enable_flip=True, flip_prob=0.5, enable_brightness=True, brightness_range=(0.8, 1.2), enable_noise=True, noise_std=0.08)
        cfg_cons = AugmentationConfig(enable_flip=True, flip_prob=0.5, enable_brightness=True, brightness_range=(0.8, 1.2), enable_noise=True, noise_std=0.08, conservative=True)
        out_non = training_preprocess(self.image, cfg_non, seed=99)
        out_cons = training_preprocess(self.image, cfg_cons, seed=99)
        img_f = self.image.astype(np.float32) / 255.0
        diff_non = np.mean(np.abs(out_non - img_f))
        diff_cons = np.mean(np.abs(out_cons - img_f))
        self.assertLess(diff_cons + 1e-8, diff_non, "Conservative should produce smaller average change")

    def test_eval_preprocess_deterministic_and_no_augmentation(self):
        # create an asymmetric image to assert eval does not flip
        img = np.zeros((4, 6, 3), dtype=np.uint8)
        img[:, :3, :] = 50
        img[:, 3:, :] = 200
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0)
        out_eval = eval_preprocess(img, cfg)
        # eval should not flip, so left-half remains 50
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

    def test_transform_toggles_and_reproducibility(self):
        # enable flip with probability 1.0 and check the result equals a manual flip
        cfg = AugmentationConfig(enable_flip=True, flip_prob=1.0, resize=(32, 32), enable_brightness=False, enable_noise=False, enable_rotate90=False)
        out = training_preprocess(self.image, cfg, seed=42)
        # manual flip of resized image (resize is no-op here)
        manual = (self.image.astype(np.float32) / 255.0)[:, ::-1, :]
        # normalization not set so compare directly
        self.assertTrue(np.allclose(out, manual, atol=1e-6))

    def test_rotate90_applied_and_reproducible(self):
        cfg = AugmentationConfig(enable_rotate90=True, rotate90_prob=1.0)
        out1 = training_preprocess(self.image, cfg, seed=7)
        out2 = training_preprocess(self.image, cfg, seed=7)
        self.assertTrue(np.array_equal(out1, out2), "Rotation should be reproducible with seed")
        # ensure eval does not perform rotation
        out_eval = eval_preprocess(self.image, cfg)
        self.assertTrue(np.array_equal(out_eval, self.image.astype(np.float32) / 255.0))

    def test_normalization(self):
        img = np.ones((4, 4, 3), dtype=np.uint8) * 128
        cfg = AugmentationConfig(normalize_mean=(0.5, 0.5, 0.5), normalize_std=(0.25, 0.25, 0.25))
        out = eval_preprocess(img, cfg)
        expected_value = (128 / 255.0 - 0.5) / 0.25
        self.assertTrue(np.allclose(out, expected_value, atol=1e-6))


if __name__ == "__main__":
    unittest.main()
