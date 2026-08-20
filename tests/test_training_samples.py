import unittest
from pathlib import Path

from src.coco.models import COCOAnnotation, COCOBoundingBox
from src.dataset.enums import Sex
from src.training.config import TrainingConfig
from src.training.samples import TrainingSample


class TestTrainingSample(unittest.TestCase):
    def _make_annotation(self, annotation_id: int, image_id: int) -> COCOAnnotation:
        return COCOAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=1,
            bbox=COCOBoundingBox(x=0.0, y=0.0, width=10.0, height=20.0),
        )

    def test_training_sample_with_side_and_rear(self) -> None:
        side = self._make_annotation(1, 101)
        rear = self._make_annotation(2, 102)

        sample = TrainingSample(
            animal_id="A-001",
            side_image=Path("side/A-001.jpg"),
            rear_image=Path("rear/A-001.jpg"),
            side_annotation=side,
            rear_annotation=rear,
            sex=Sex.MALE,
            weight_kg=120.5,
        )

        self.assertEqual(sample.animal_id, "A-001")
        self.assertEqual(sample.side_image, Path("side/A-001.jpg"))
        self.assertEqual(sample.rear_image, Path("rear/A-001.jpg"))
        self.assertIs(sample.side_annotation, side)
        self.assertIs(sample.rear_annotation, rear)
        self.assertEqual(sample.sex, Sex.MALE)
        self.assertEqual(sample.weight_kg, 120.5)

    def test_training_sample_with_side_only(self) -> None:
        sample = TrainingSample(
            animal_id="B-002",
            side_image=Path("side/B-002.jpg"),
            rear_image=None,
            side_annotation=self._make_annotation(3, 103),
            rear_annotation=None,
            sex=Sex.FEMALE,
            weight_kg=140.0,
        )

        self.assertEqual(sample.side_image, Path("side/B-002.jpg"))
        self.assertIsNone(sample.rear_image)
        self.assertIsNotNone(sample.side_annotation)
        self.assertIsNone(sample.rear_annotation)
        self.assertEqual(sample.sex, Sex.FEMALE)

    def test_training_sample_with_rear_only(self) -> None:
        sample = TrainingSample(
            animal_id="C-003",
            side_image=None,
            rear_image=Path("rear/C-003.jpg"),
            side_annotation=None,
            rear_annotation=self._make_annotation(4, 104),
            sex=Sex.MALE,
            weight_kg=110.75,
        )

        self.assertIsNone(sample.side_image)
        self.assertEqual(sample.rear_image, Path("rear/C-003.jpg"))
        self.assertIsNone(sample.side_annotation)
        self.assertIsNotNone(sample.rear_annotation)
        self.assertEqual(sample.weight_kg, 110.75)

    def test_training_config_preserves_explicit_values(self) -> None:
        config = TrainingConfig(
            seed=42,
            input_size=(224, 224),
            batch_size=16,
            epochs=5,
            learning_rate=0.001,
            optimizer="adamw",
            scheduler="cosine",
            weight_decay=0.0001,
            device="cpu",
            output_directory=Path("runs/test"),
        )

        self.assertEqual(config.seed, 42)
        self.assertEqual(config.input_size, (224, 224))
        self.assertEqual(config.batch_size, 16)
        self.assertEqual(config.epochs, 5)
        self.assertEqual(config.learning_rate, 0.001)
        self.assertEqual(config.optimizer, "adamw")
        self.assertEqual(config.scheduler, "cosine")
        self.assertEqual(config.weight_decay, 0.0001)
        self.assertEqual(config.device, "cpu")
        self.assertEqual(config.output_directory, Path("runs/test"))


if __name__ == "__main__":
    unittest.main()
