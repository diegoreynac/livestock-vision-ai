import unittest
from pathlib import Path

from src.analysis.dataset_identity_audit import DatasetIdentityAudit
from src.dataset.enums import DatasetType, Sex, View
from src.dataset.models import ImageFolder, ImageRecord


class TestDatasetIdentityAudit(unittest.TestCase):
    def setUp(self) -> None:
        self.side_b2 = ImageFolder(
            dataset=DatasetType.B2,
            folder_name="Side",
            view=View.SIDE,
            path=Path("dummy_side_b2"),
        )
        self.rear_b2 = ImageFolder(
            dataset=DatasetType.B2,
            folder_name="Rear",
            view=View.REAR,
            path=Path("dummy_rear_b2"),
        )
        self.side_b4 = ImageFolder(
            dataset=DatasetType.B4,
            folder_name="Side",
            view=View.SIDE,
            path=Path("dummy_side_b4"),
        )
        self.rear_b4 = ImageFolder(
            dataset=DatasetType.B4,
            folder_name="Rear",
            view=View.REAR,
            path=Path("dummy_rear_b4"),
        )
        self.side_b3 = ImageFolder(
            dataset=DatasetType.B3,
            folder_name="Side",
            view=View.SIDE,
            path=Path("dummy_side_b3"),
        )

    def test_audit_counts_invalid_ids_and_dataset_membership(self) -> None:
        records = [
            ImageRecord(animal_id="A", weight_kg=100.0, sex=Sex.MALE, filename="a1.jpg", filepath=Path("a1.jpg"), folder=self.side_b2),
            ImageRecord(animal_id="A", weight_kg=100.0, sex=Sex.MALE, filename="a2.jpg", filepath=Path("a2.jpg"), folder=self.rear_b4),
            ImageRecord(animal_id="B", weight_kg=130.0, sex=Sex.FEMALE, filename="b1.jpg", filepath=Path("b1.jpg"), folder=self.side_b3),
            ImageRecord(animal_id="", weight_kg=140.0, sex=Sex.FEMALE, filename="bad.jpg", filepath=Path("bad.jpg"), folder=self.side_b2),
        ]

        report = DatasetIdentityAudit().audit(records)

        self.assertEqual(report["total_images"], 4)
        self.assertEqual(report["invalid_animal_id_records"], 1)
        self.assertEqual(report["total_valid_animal_ids"], 2)
        self.assertEqual(report["dataset_membership"]["B2 + B4"], 1)
        self.assertEqual(report["dataset_membership"]["B3 only"], 1)

    def test_audit_reports_view_pairing_and_consistency(self) -> None:
        paired_side = ImageRecord(animal_id="C", weight_kg=120.0, sex=Sex.MALE, filename="c_side.jpg", filepath=Path("c_side.jpg"), folder=self.side_b2)
        paired_rear = ImageRecord(animal_id="C", weight_kg=120.0, sex=Sex.MALE, filename="c_rear.jpg", filepath=Path("c_rear.jpg"), folder=self.rear_b4)
        single_side = ImageRecord(animal_id="D", weight_kg=111.0, sex=Sex.FEMALE, filename="d_side.jpg", filepath=Path("d_side.jpg"), folder=self.side_b4)

        report = DatasetIdentityAudit().audit([paired_side, paired_rear, single_side])

        self.assertEqual(report["view_membership"]["Side + Rear"], 1)
        self.assertEqual(report["view_membership"]["Side only"], 1)
        self.assertEqual(report["side_rear_pairing"]["animals_with_both_views_count"], 1)
        self.assertEqual(report["weight_consistency"]["consistent_animals"], ["C", "D"])
        self.assertEqual(report["sex_consistency"]["consistent_animals"], ["C", "D"])
        self.assertIn("C", [item["animal_id"] for item in report["cross_dataset_identity_risk"]["cross_dataset_animals"]])


if __name__ == "__main__":
    unittest.main()
