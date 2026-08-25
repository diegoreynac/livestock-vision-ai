import unittest
from pathlib import Path

from src.analysis.animal_analysis import AnimalAnalysis
from src.coco.models import COCOAnnotation, COCOBoundingBox, COCOKeypoint
from src.coco.enums import KeypointVisibility
from src.dataset.enums import DatasetType, Sex, View
from src.dataset.models import ImageFolder, ImageRecord


class TestAnimalAnalysis(unittest.TestCase):
    def setUp(self) -> None:
        self.side_folder = ImageFolder(
            dataset=DatasetType.B2,
            folder_name="Side",
            view=View.SIDE,
            path=Path("dummy_side"),
        )
        self.rear_folder = ImageFolder(
            dataset=DatasetType.B4,
            folder_name="Rear",
            view=View.REAR,
            path=Path("dummy_rear"),
        )

    def test_weight_consistency_within_tolerance(self) -> None:
        records = [
            ImageRecord(animal_id="A", weight_kg=100.0, sex=Sex.MALE, filename="a1.jpg", filepath=Path("a1.jpg")),
            ImageRecord(animal_id="A", weight_kg=100.005, sex=Sex.MALE, filename="a2.jpg", filepath=Path("a2.jpg")),
        ]
        for record in records:
            record.folder = self.side_folder

        report = AnimalAnalysis(weight_tolerance=0.01).analyze(records)

        self.assertEqual(report.total_animals, 1)
        self.assertEqual(report.animals[0].consistent_weight, True)
        self.assertEqual(report.weight_distribution, {"100-150": 1})

    def test_weight_inconsistency_above_tolerance(self) -> None:
        records = [
            ImageRecord(animal_id="A", weight_kg=100.0, sex=Sex.MALE, filename="a1.jpg", filepath=Path("a1.jpg")),
            ImageRecord(animal_id="A", weight_kg=100.1, sex=Sex.MALE, filename="a2.jpg", filepath=Path("a2.jpg")),
        ]
        for record in records:
            record.folder = self.side_folder

        report = AnimalAnalysis(weight_tolerance=0.01).analyze(records)

        self.assertEqual(report.total_animals, 1)
        self.assertFalse(report.animals[0].consistent_weight)
        self.assertEqual(report.weight_distribution, {"inconsistent": 1})

    def test_sex_consistency_preserves_observed_values(self) -> None:
        records = [
            ImageRecord(animal_id="B", weight_kg=120.0, sex=Sex.FEMALE, filename="b1.jpg", filepath=Path("b1.jpg")),
            ImageRecord(animal_id="B", weight_kg=120.0, sex=Sex.MALE, filename="b2.jpg", filepath=Path("b2.jpg")),
        ]
        for record in records:
            record.folder = self.side_folder

        report = AnimalAnalysis().analyze(records)

        self.assertEqual(report.total_animals, 1)
        self.assertFalse(report.animals[0].consistent_sex)
        self.assertEqual(report.animals[0].sex_values, ["F", "M"])
        self.assertEqual(report.sex_distribution, {"inconsistent": 1})

    def test_dataset_membership_distribution_and_exclusive(self) -> None:
        record_b2 = ImageRecord(animal_id="C", weight_kg=130.0, sex=Sex.FEMALE, filename="c1.jpg", filepath=Path("c1.jpg"))
        record_b4 = ImageRecord(animal_id="C", weight_kg=130.0, sex=Sex.FEMALE, filename="c2.jpg", filepath=Path("c2.jpg"))
        record_b2.folder = self.side_folder
        record_b4.folder = self.rear_folder

        report = AnimalAnalysis().analyze([record_b2, record_b4])

        self.assertEqual(report.total_animals, 1)
        self.assertEqual(report.dataset_membership_distribution, {"B2": 1, "B4": 1})
        self.assertEqual(report.dataset_exclusive_distribution, {"B2 + B4": 1})

    def test_side_plus_rear_view_category(self) -> None:
        side_record = ImageRecord(animal_id="D", weight_kg=140.0, sex=Sex.MALE, filename="d1.jpg", filepath=Path("d1.jpg"))
        rear_record = ImageRecord(animal_id="D", weight_kg=140.0, sex=Sex.MALE, filename="d2.jpg", filepath=Path("d2.jpg"))
        side_record.folder = self.side_folder
        rear_record.folder = self.rear_folder

        report = AnimalAnalysis().analyze([side_record, rear_record])

        self.assertEqual(report.view_distribution, {"Side + Rear": 1})
        self.assertEqual(report.animals[0].view_category, "Side + Rear")

    def test_record_without_folder_does_not_crash(self) -> None:
        no_folder_record = ImageRecord(animal_id="E", weight_kg=160.0, sex=Sex.MALE, filename="e1.jpg", filepath=Path("e1.jpg"))

        report = AnimalAnalysis().analyze([no_folder_record])

        self.assertEqual(report.total_animals, 1)
        self.assertEqual(report.animals[0].dataset_types, ["Unknown"])
        self.assertEqual(report.animals[0].view_types, ["Unknown"])
        self.assertEqual(report.records_with_missing_folder, 1)

    def test_record_without_weight_does_not_crash(self) -> None:
        no_weight_record = ImageRecord(animal_id="F", weight_kg=None, sex=Sex.FEMALE, filename="f1.jpg", filepath=Path("f1.jpg"))
        no_weight_record.folder = self.side_folder

        report = AnimalAnalysis().analyze([no_weight_record])

        self.assertEqual(report.total_animals, 1)
        self.assertEqual(report.records_with_missing_weight, 1)
        self.assertEqual(report.weight_distribution, {"missing": 1})

    def test_record_without_sex_does_not_crash(self) -> None:
        no_sex_record = ImageRecord(animal_id="G", weight_kg=170.0, sex=None, filename="g1.jpg", filepath=Path("g1.jpg"))
        no_sex_record.folder = self.side_folder

        report = AnimalAnalysis().analyze([no_sex_record])

        self.assertEqual(report.total_animals, 1)
        self.assertEqual(report.records_with_missing_sex, 1)
        self.assertEqual(report.sex_distribution, {"missing": 1})

    def test_empty_animal_id_is_excluded_and_reported(self) -> None:
        empty_id_record = ImageRecord(animal_id="", weight_kg=180.0, sex=Sex.MALE, filename="h1.jpg", filepath=Path("h1.jpg"))
        empty_id_record.folder = self.side_folder

        report = AnimalAnalysis().analyze([empty_id_record])

        self.assertEqual(report.total_animals, 0)
        self.assertEqual(report.invalid_animal_id_records, 1)
        self.assertEqual(report.total_images, 1)
        self.assertEqual(report.valid_image_count, 0)


if __name__ == "__main__":
    unittest.main()
