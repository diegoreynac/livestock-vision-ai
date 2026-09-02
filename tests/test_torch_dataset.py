import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch

from src.coco.models import COCOAnnotation, COCOBoundingBox
from src.dataset.enums import Sex
from src.training.augmentation import (
    AugmentationConfig,
    training_preprocess as original_training_preprocess,
)
from src.training.samples import TrainingSample
from src.training.torch_dataset import InputMode, LivestockDataset


class TestTorchLivestockDataset(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.side_path = root / "side.png"
        self.rear_path = root / "rear.png"
        cv2.imwrite(str(self.side_path), np.full((4, 6, 3), (30, 20, 10), dtype=np.uint8))
        cv2.imwrite(str(self.rear_path), np.full((4, 6, 3), (60, 50, 40), dtype=np.uint8))
        self.sample = TrainingSample(
            animal_id=" exact-id ",
            side_image=self.side_path,
            rear_image=self.rear_path,
            side_annotation=self._annotation(1, 1, 1, 2, 2),
            rear_annotation=self._annotation(2, 1, 1, 2, 2),
            sex=Sex.MALE,
            weight_kg=123.5,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _annotation(annotation_id, x, y, width, height):
        return COCOAnnotation(
            id=annotation_id,
            image_id=annotation_id,
            category_id=1,
            bbox=COCOBoundingBox(x=x, y=y, width=width, height=height),
        )

    def _dataset(self, mode=InputMode.SIDE, training=False, samples=None):
        return LivestockDataset(
            [self.sample] if samples is None else samples,
            input_mode=mode,
            training=training,
            augmentation_config=AugmentationConfig(),
        )

    def test_length_and_side_sample(self):
        dataset = self._dataset()
        result = dataset[0]
        self.assertEqual(len(dataset), 1)
        self.assertEqual(set(result), {"image", "bbox", "weight", "animal_id"})
        self.assertEqual(result["image"].shape, (3, 4, 6))
        self.assertEqual(result["bbox"].shape, (4,))
        self.assertTrue(torch.all((result["bbox"] >= 0) & (result["bbox"] <= 1)))
        self.assertEqual(result["weight"].shape, (1,))
        self.assertEqual(result["weight"].dtype, torch.float32)
        self.assertEqual(result["weight"].item(), 123.5)
        self.assertEqual(result["animal_id"], " exact-id ")

    def test_invalid_input_mode_fails_clearly(self):
        with self.assertRaisesRegex(ValueError, "Unsupported input mode"):
            self._dataset("diagonal")

    def test_rear_sample(self):
        result = self._dataset(InputMode.REAR)[0]
        self.assertEqual(result["image"].shape, (3, 4, 6))
        self.assertTrue(torch.allclose(
            result["bbox"], torch.tensor([1 / 6, 1 / 4, 3 / 6, 3 / 4])
        ))

    def test_side_rear_channel_order(self):
        image = self._dataset(InputMode.SIDE_REAR)[0]["image"]
        self.assertEqual(image.shape, (6, 4, 6))
        self.assertTrue(torch.allclose(image[:3, 0, 0], torch.tensor([10, 20, 30]) / 255))
        self.assertTrue(torch.allclose(image[3:, 0, 0], torch.tensor([40, 50, 60]) / 255))

    def test_side_rear_geometric_transform_is_consistent(self):
        sample = TrainingSample(
            "paired",
            self.side_path,
            self.rear_path,
            self._annotation(1, 1, 0, 2, 2),
            self._annotation(2, 3, 1, 1, 2),
            Sex.MALE,
            100.0,
        )
        config = AugmentationConfig(
            resize=(8, 12),
            enable_flip=True,
            flip_prob=1.0,
            enable_rotation=True,
            rotation_range=(10.0, 10.0),
            rotation_prob=1.0,
            enable_zoom=True,
            zoom_range=(1.1, 1.1),
            zoom_prob=1.0,
        )
        transformed_boxes = []
        seeds = []

        def preprocess(image, config, seed=None, annotations=None):
            seeds.append(seed)
            result = original_training_preprocess(
                image,
                config,
                seed=seed,
                annotations=annotations,
            )
            transformed_boxes.append(annotations["boxes"][0].copy())
            return result

        with patch("src.training.torch_dataset.training_preprocess", preprocess):
            dataset = LivestockDataset(
                [sample],
                input_mode=InputMode.SIDE_REAR,
                training=True,
                augmentation_config=config,
            )
            result = dataset[0]

        self.assertEqual(result["image"].shape, (6, 8, 12))
        self.assertEqual(len(transformed_boxes), 2)
        self.assertEqual(seeds, [0, 0])
        expected_boxes = []
        for box in ([1, 0, 3, 2], [3, 1, 4, 3]):
            annotations = {"boxes": np.array([box], dtype=np.float32)}
            original_training_preprocess(
                np.zeros((4, 6, 3), dtype=np.uint8),
                config,
                seed=0,
                annotations=annotations,
            )
            expected_boxes.append(annotations["boxes"][0])
        self.assertTrue(np.allclose(
            transformed_boxes,
            expected_boxes,
            atol=1e-5,
        ))

    def test_coco_conversion_and_clipping(self):
        sample = TrainingSample(
            "a", self.side_path, None, self._annotation(1, 5, 3, 1, 1), None, Sex.MALE, 1
        )
        def preprocess(image, config, annotations=None):
            annotations["boxes"][0] = [5, 3, 8, 6]
            return image

        with patch("src.training.torch_dataset.eval_preprocess", preprocess):
            result = self._dataset(samples=[sample])[0]
        self.assertTrue(torch.allclose(
            result["bbox"], torch.tensor([5 / 6, 3 / 4, 1.0, 1.0])
        ))

    def test_transformed_bbox_is_clipped_before_normalization(self):
        def preprocess(image, config, seed=None, annotations=None):
            annotations["boxes"][0] = [-2, 1, 8, 6]
            return image

        with patch("src.training.torch_dataset.training_preprocess", preprocess):
            result = self._dataset(training=True)[0]
        self.assertEqual(result["bbox"].tolist(), [0.0, 1 / 4, 1.0, 1.0])

    def test_training_and_evaluation_paths(self):
        with patch("src.training.torch_dataset.eval_preprocess", wraps=lambda image, config, annotations=None: image) as evaluation:
            self._dataset(training=False)[0]
        evaluation.assert_called_once()

        with patch("src.training.torch_dataset.training_preprocess", wraps=lambda image, config, seed=None, annotations=None: image) as training:
            self._dataset(training=True)[0]
        training.assert_called_once()

    def test_missing_side_rear_inputs_fail_clearly(self):
        with self.assertRaisesRegex(ValueError, "Missing Side image"):
            self._dataset(InputMode.SIDE_REAR, samples=[self.sample.__class__(
                self.sample.animal_id, None, self.rear_path, self.sample.side_annotation,
                self.sample.rear_annotation, self.sample.sex, self.sample.weight_kg
            )])[0]
        with self.assertRaisesRegex(ValueError, "Missing Rear image"):
            self._dataset(InputMode.SIDE_REAR, samples=[self.sample.__class__(
                self.sample.animal_id, self.side_path, None, self.sample.side_annotation,
                self.sample.rear_annotation, self.sample.sex, self.sample.weight_kg
            )])[0]

    def test_missing_bbox_fails_without_keypoint_derivation(self):
        sample = self.sample.__class__(
            self.sample.animal_id, self.side_path, self.rear_path, None,
            self.sample.rear_annotation, self.sample.sex, self.sample.weight_kg
        )
        with self.assertRaisesRegex(ValueError, "Missing valid Side COCO bounding box"):
            self._dataset(samples=[sample])[0]

    def test_images_are_loaded_lazily(self):
        with patch("src.training.torch_dataset.cv2.imread", side_effect=AssertionError):
            dataset = self._dataset()
        self.assertEqual(len(dataset), 1)


if __name__ == "__main__":
    unittest.main()
