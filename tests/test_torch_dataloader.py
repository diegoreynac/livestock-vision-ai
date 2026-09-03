import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import torch

from src.coco.models import COCOAnnotation, COCOBoundingBox
from src.dataset.enums import Sex
from src.training.augmentation import AugmentationConfig
from src.training.samples import TrainingSample
from src.training.torch_dataloader import create_dataloader
from src.training.torch_dataset import InputMode, LivestockDataset


class TestTorchDataLoader(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        root = Path(self.temp_dir.name)
        self.samples = []
        for index in range(5):
            side_path = root / f"side-{index}.png"
            rear_path = root / f"rear-{index}.png"
            cv2.imwrite(
                str(side_path),
                np.full((4, 6, 3), (30 + index, 20 + index, 10 + index), dtype=np.uint8),
            )
            cv2.imwrite(
                str(rear_path),
                np.full((4, 6, 3), (60 + index, 50 + index, 40 + index), dtype=np.uint8),
            )
            self.samples.append(
                TrainingSample(
                    animal_id=f"animal-{index}",
                    side_image=side_path,
                    rear_image=rear_path,
                    side_annotation=self._annotation(index),
                    rear_annotation=self._annotation(index),
                    sex=Sex.MALE,
                    weight_kg=100.0 + index,
                )
            )

    def tearDown(self):
        self.temp_dir.cleanup()

    @staticmethod
    def _annotation(annotation_id: int) -> COCOAnnotation:
        return COCOAnnotation(
            id=annotation_id,
            image_id=annotation_id,
            category_id=1,
            bbox=COCOBoundingBox(x=1, y=1, width=2, height=2),
        )

    def _dataset(self, mode=InputMode.SIDE, samples=None):
        return LivestockDataset(
            self.samples if samples is None else samples,
            input_mode=mode,
            training=False,
            augmentation_config=AugmentationConfig(),
        )

    def test_creation_returns_torch_dataloader(self):
        loader = create_dataloader(self._dataset(), batch_size=2)

        self.assertIsInstance(loader, torch.utils.data.DataLoader)

    def test_side_batch_shapes_and_ids(self):
        batch = next(iter(create_dataloader(self._dataset(), batch_size=2)))

        self.assertEqual(batch["image"].shape, (2, 3, 4, 6))
        self.assertEqual(batch["bbox"].shape, (2, 4))
        self.assertEqual(batch["weight"].shape, (2, 1))
        self.assertEqual(batch["animal_id"], ["animal-0", "animal-1"])

    def test_rear_batch_shape(self):
        batch = next(
            iter(create_dataloader(self._dataset(InputMode.REAR), batch_size=2))
        )

        self.assertEqual(batch["image"].shape, (2, 3, 4, 6))

    def test_side_rear_batch_preserves_channel_order(self):
        batch = next(
            iter(create_dataloader(self._dataset(InputMode.SIDE_REAR), batch_size=2))
        )

        self.assertEqual(batch["image"].shape, (2, 6, 4, 6))
        self.assertTrue(
            torch.allclose(batch["image"][0, :3, 0, 0], torch.tensor([10, 20, 30]) / 255)
        )
        self.assertTrue(
            torch.allclose(batch["image"][0, 3:, 0, 0], torch.tensor([40, 50, 60]) / 255)
        )

    def test_shuffle_false_preserves_sample_order(self):
        loader = create_dataloader(self._dataset(), batch_size=2, shuffle=False)

        ids = [animal_id for batch in loader for animal_id in batch["animal_id"]]

        self.assertEqual(ids, ["animal-0", "animal-1", "animal-2", "animal-3", "animal-4"])

    def test_drop_last_controls_incomplete_batch(self):
        keep_loader = create_dataloader(self._dataset(), batch_size=2, drop_last=False)
        drop_loader = create_dataloader(self._dataset(), batch_size=2, drop_last=True)

        self.assertEqual(len(list(keep_loader)), 3)
        self.assertEqual(len(list(drop_loader)), 2)

    def test_num_workers_is_configured_without_worker_iteration(self):
        loader = create_dataloader(self._dataset(), num_workers=1)

        self.assertEqual(loader.num_workers, 1)

    def test_dataset_errors_propagate(self):
        invalid_sample = TrainingSample(
            animal_id="invalid",
            side_image=None,
            rear_image=self.samples[0].rear_image,
            side_annotation=self.samples[0].side_annotation,
            rear_annotation=self.samples[0].rear_annotation,
            sex=Sex.MALE,
            weight_kg=100.0,
        )

        with self.assertRaisesRegex(ValueError, "Missing Side image"):
            next(iter(create_dataloader(self._dataset(samples=[invalid_sample]))))


if __name__ == "__main__":
    unittest.main()
