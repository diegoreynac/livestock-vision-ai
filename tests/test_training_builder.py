import unittest
from pathlib import Path

from src.coco.models import COCOAnnotation, COCOBoundingBox
from src.dataset.enums import DatasetType, Sex, View
from src.dataset.models import ImageFolder, ImageRecord
from src.training.builder import DatasetBuilder


class TestDatasetBuilder(unittest.TestCase):
    def _make_annotation(self, annotation_id: int, image_id: int) -> COCOAnnotation:
        return COCOAnnotation(
            id=annotation_id,
            image_id=image_id,
            category_id=1,
            bbox=COCOBoundingBox(x=0.0, y=0.0, width=10.0, height=20.0),
        )

    def _make_record(
        self,
        animal_id: str,
        filename: str,
        view: View,
        dataset: DatasetType,
        annotation=None,
        weight=100.0,
        sex=Sex.MALE,
    ):
        folder = ImageFolder(dataset=dataset, folder_name=f"f-{dataset.value}", view=view, path=Path("/tmp"))
        record = ImageRecord(
            animal_id=animal_id,
            weight_kg=weight,
            sex=sex,
            filename=filename,
            filepath=Path(filename),
            extra=None,
            folder=folder,
            annotation=annotation,
        )
        folder.records.append(record)
        return record

    def test_pairing_prefers_annotation_and_handles_multiple_per_view(self):
        side_annot = self._make_annotation(1, 101)
        rear_annot = self._make_annotation(2, 102)

        s1 = self._make_record("A", "a_side_1.jpg", View.SIDE, DatasetType.B2, annotation=None)
        s2 = self._make_record("A", "a_side_2.jpg", View.SIDE, DatasetType.B2, annotation=side_annot)
        r1 = self._make_record("A", "a_rear_1.jpg", View.REAR, DatasetType.B2, annotation=rear_annot)
        r2 = self._make_record("A", "a_rear_2.jpg", View.REAR, DatasetType.B2, annotation=None)

        builder = DatasetBuilder()
        samples = builder.build([s1, s2, r1, r2])

        self.assertEqual(len(samples), 1)
        sample = samples[0]
        self.assertEqual(sample.side_annotation, side_annot)
        self.assertEqual(sample.rear_annotation, rear_annot)

    def test_dual_view_filtering(self):
        s = self._make_record("B", "b_side.jpg", View.SIDE, DatasetType.B3, annotation=None)
        c_s = self._make_record("C", "c_side.jpg", View.SIDE, DatasetType.B3, annotation=None)
        c_r = self._make_record("C", "c_rear.jpg", View.REAR, DatasetType.B3, annotation=None)

        builder = DatasetBuilder()
        samples = builder.build([s, c_s, c_r])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].animal_id, "C")

    def test_missing_sex_is_rejected(self):
        s = self._make_record("D", "d_side.jpg", View.SIDE, DatasetType.B4, annotation=None, sex=None)
        r = self._make_record("D", "d_rear.jpg", View.REAR, DatasetType.B4, annotation=None, sex=None)

        builder = DatasetBuilder()
        with self.assertRaises(ValueError):
            builder.build([s, r])

    def test_inconsistent_sex_and_weight_are_rejected(self):
        d1 = self._make_record("E", "e_side.jpg", View.SIDE, DatasetType.B4, annotation=None, weight=100.0, sex=Sex.MALE)
        d2 = self._make_record("E", "e_rear.jpg", View.REAR, DatasetType.B4, annotation=None, weight=110.0, sex=Sex.FEMALE)

        builder = DatasetBuilder()
        with self.assertRaises(ValueError):
            builder.build([d1, d2])

        e1 = self._make_record("F", "f_side.jpg", View.SIDE, DatasetType.B4, annotation=None, weight=100.0, sex=Sex.MALE)
        e2 = self._make_record("F", "f_rear.jpg", View.REAR, DatasetType.B4, annotation=None, weight=100.0, sex=Sex.MALE)
        self.assertEqual(len(builder.build([e1, e2])), 1)

    def test_invalid_animal_id_values_are_excluded(self):
        valid = self._make_record("G", "g_side.jpg", View.SIDE, DatasetType.B4, annotation=None, sex=Sex.MALE)
        valid_rear = self._make_record("G", "g_rear.jpg", View.REAR, DatasetType.B4, annotation=None, sex=Sex.MALE)
        invalid_none = self._make_record(None, "none_side.jpg", View.SIDE, DatasetType.B4, annotation=None, sex=Sex.MALE)
        invalid_empty = self._make_record("", "empty_side.jpg", View.SIDE, DatasetType.B4, annotation=None, sex=Sex.MALE)
        invalid_whitespace = self._make_record("   ", "whitespace_side.jpg", View.SIDE, DatasetType.B4, annotation=None, sex=Sex.MALE)

        builder = DatasetBuilder()
        samples = builder.build([valid, valid_rear, invalid_none, invalid_empty, invalid_whitespace])

        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].animal_id, "G")

    def test_same_records_in_different_orders_produce_same_result(self):
        side_annot = self._make_annotation(7, 701)
        rear_annot = self._make_annotation(8, 802)

        first_order = [
            self._make_record("H", "h_side_2.jpg", View.SIDE, DatasetType.B2, annotation=side_annot),
            self._make_record("H", "h_side_1.jpg", View.SIDE, DatasetType.B2, annotation=None),
            self._make_record("H", "h_rear_2.jpg", View.REAR, DatasetType.B2, annotation=None),
            self._make_record("H", "h_rear_1.jpg", View.REAR, DatasetType.B2, annotation=rear_annot),
        ]
        second_order = list(reversed(first_order))

        builder = DatasetBuilder()
        left = builder.build(first_order)
        right = builder.build(second_order)

        self.assertEqual(len(left), 1)
        self.assertEqual(len(right), 1)
        self.assertEqual(left[0].animal_id, right[0].animal_id)
        self.assertEqual(left[0].side_annotation, right[0].side_annotation)
        self.assertEqual(left[0].rear_annotation, right[0].rear_annotation)


if __name__ == "__main__":
    unittest.main()
