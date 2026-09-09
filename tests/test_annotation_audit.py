"""
Tests for the COCO annotation quality audit (module 6/7 of the audit).

All tests use small synthetic COCO JSON structures. No real dataset image
files or JSON exports are required.
"""

from __future__ import annotations

import copy
import json
import unittest
from pathlib import Path

from src.analysis.annotation_quality_audit import (
    AnimalPairingStatus,
    AnnotationQualityAuditor,
)
from src.analysis.annotation_quality_report import AnnotationQualityReport
from src.coco.enums import AnnotationStatus
from src.dataset.enums import DatasetType, Sex, View


def _coco(images: list[dict], annotations: list[dict]) -> dict:

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cow"}],
    }


def _image(image_id: int, file_name: str, width=1920, height=1080) -> dict:

    return {
        "id": image_id,
        "file_name": file_name,
        "width": width,
        "height": height,
    }


def _annotation(ann_id: int, image_id: int, bbox) -> dict:

    return {
        "id": ann_id,
        "image_id": image_id,
        "category_id": 1,
        "bbox": bbox,
    }


class TestBBoxStatusClassification(unittest.TestCase):
    """Covers cases 1-11 of the task's minimum test list."""

    def setUp(self) -> None:
        self.auditor = AnnotationQualityAuditor()

    def _audit_one(self, bbox, width=1920, height=1080):

        data = _coco(
            images=[_image(1, "30_b4-1_s_110_F.jpg", width, height)],
            annotations=[_annotation(1, 1, bbox)],
        )

        report = self.auditor.audit_json(data, DatasetType.B4, View.SIDE)

        return report.images[0]

    def test_valid_bbox(self) -> None:
        image = self._audit_one([10, 10, 100, 200])
        self.assertEqual(image.status, AnnotationStatus.VALID)
        self.assertEqual(image.annotations[0].bbox_xyxy, (10, 10, 110, 210))

    def test_missing_annotation(self) -> None:
        data = _coco(
            images=[_image(1, "30_b4-1_s_110_F.jpg")],
            annotations=[],
        )
        report = self.auditor.audit_json(data, DatasetType.B4, View.SIDE)
        self.assertEqual(report.images[0].status, AnnotationStatus.MISSING)
        self.assertEqual(report.images[0].annotation_count, 0)

    def test_zero_bbox(self) -> None:
        image = self._audit_one([0, 0, 0, 0])
        self.assertEqual(image.status, AnnotationStatus.ZERO_BBOX)

    def test_zero_width(self) -> None:
        image = self._audit_one([10, 10, 0, 50])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_zero_height(self) -> None:
        image = self._audit_one([10, 10, 50, 0])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_negative_width(self) -> None:
        image = self._audit_one([10, 10, -50, 50])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_negative_height(self) -> None:
        image = self._audit_one([10, 10, 50, -50])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_nan_bbox(self) -> None:
        image = self._audit_one([float("nan"), 10, 50, 50])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_inf_bbox(self) -> None:
        image = self._audit_one([10, 10, float("inf"), 50])
        self.assertEqual(image.status, AnnotationStatus.INVALID_BBOX)

    def test_out_of_bounds(self) -> None:
        image = self._audit_one([1900, 1000, 100, 200], width=1920, height=1080)
        self.assertEqual(image.status, AnnotationStatus.OUT_OF_BOUNDS)

    def test_bbox_touching_boundary_is_valid(self) -> None:
        # x + w == width, y + h == height exactly -> still VALID.
        image = self._audit_one([1820, 880, 100, 200], width=1920, height=1080)
        self.assertEqual(image.status, AnnotationStatus.VALID)

    def test_multiple_annotations(self) -> None:
        data = _coco(
            images=[_image(1, "30_b4-1_s_110_F.jpg")],
            annotations=[
                _annotation(1, 1, [10, 10, 50, 50]),
                _annotation(2, 1, [20, 20, 60, 60]),
            ],
        )
        report = self.auditor.audit_json(data, DatasetType.B4, View.SIDE)
        image = report.images[0]
        self.assertEqual(image.status, AnnotationStatus.MULTIPLE_ANNOTATIONS)
        self.assertEqual(image.annotation_count, 2)
        # Both annotations preserved, none discarded.
        self.assertEqual(
            {a.annotation_id for a in image.annotations}, {1, 2}
        )


class TestAnimalLevelPairing(unittest.TestCase):
    """Covers cases 13-17."""

    def setUp(self) -> None:
        self.auditor = AnnotationQualityAuditor()

    def test_valid_side_and_valid_rear(self) -> None:
        side_data = _coco(
            images=[_image(1, "30_b4-1_s_110_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, 100, 100])],
        )
        rear_data = _coco(
            images=[_image(1, "30_b4-1_r_110_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, 100, 100])],
        )

        report = self.auditor.audit([
            (side_data, DatasetType.B4, View.SIDE),
            (rear_data, DatasetType.B4, View.REAR),
        ])

        animal = report.animals[0]
        self.assertEqual(animal.animal_id, "30_b4-1")
        self.assertEqual(
            animal.pairing_status, AnimalPairingStatus.SIDE_REAR_VALID
        )

    def test_valid_side_missing_rear(self) -> None:
        side_data = _coco(
            images=[_image(1, "31_b4-1_s_110_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, 100, 100])],
        )
        rear_data = _coco(images=[], annotations=[])

        report = self.auditor.audit([
            (side_data, DatasetType.B4, View.SIDE),
            (rear_data, DatasetType.B4, View.REAR),
        ])

        animal = report.animals[0]
        self.assertEqual(animal.pairing_status, AnimalPairingStatus.SIDE_ONLY)

    def test_missing_side_valid_rear(self) -> None:
        side_data = _coco(images=[], annotations=[])
        rear_data = _coco(
            images=[_image(1, "32_b4-2_r_110_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, 100, 100])],
        )

        report = self.auditor.audit([
            (side_data, DatasetType.B4, View.SIDE),
            (rear_data, DatasetType.B4, View.REAR),
        ])

        animal = report.animals[0]
        self.assertEqual(animal.pairing_status, AnimalPairingStatus.REAR_ONLY)

    def test_neither_view_valid(self) -> None:
        side_data = _coco(
            images=[_image(1, "33_b4-1_s_110_F.jpg")],
            annotations=[_annotation(1, 1, [0, 0, 0, 0])],
        )
        rear_data = _coco(
            images=[_image(2, "33_b4-2_r_110_F.jpg")],
            annotations=[],
        )

        report = self.auditor.audit([
            (side_data, DatasetType.B4, View.SIDE),
            (rear_data, DatasetType.B4, View.REAR),
        ])

        animal = report.animals[0]
        self.assertEqual(
            animal.pairing_status, AnimalPairingStatus.NO_VALID_VIEW
        )

    def test_deterministic_ordering(self) -> None:
        side_data = _coco(
            images=[
                _image(2, "5_b4-1_s_100_F.jpg"),
                _image(1, "1_b4-1_s_100_F.jpg"),
            ],
            annotations=[
                _annotation(1, 2, [10, 10, 50, 50]),
                _annotation(2, 1, [10, 10, 50, 50]),
            ],
        )

        report = self.auditor.audit_json(side_data, DatasetType.B4, View.SIDE)

        file_names = [image.file_name for image in report.sorted_images]
        self.assertEqual(
            file_names, ["1_b4-1_s_100_F.jpg", "5_b4-1_s_100_F.jpg"]
        )

        animal_ids = [animal.animal_id for animal in report.sorted_animals]
        self.assertEqual(animal_ids, sorted(animal_ids))


class TestReportGeneration(unittest.TestCase):
    """Covers cases 18-20 plus non-mutation guarantees."""

    def setUp(self) -> None:
        self.auditor = AnnotationQualityAuditor()

        self.side_data = _coco(
            images=[_image(1, "30_b4-1_s_110_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, 100, 200])],
        )
        self.rear_data = _coco(
            images=[_image(1, "31_b4-2_r_117_F.jpg")],
            annotations=[],
        )

        self.report = self.auditor.audit([
            (self.side_data, DatasetType.B4, View.SIDE),
            (self.rear_data, DatasetType.B4, View.REAR),
        ])

    def test_markdown_contains_animal_level_information(self) -> None:
        markdown = AnnotationQualityReport(self.report).to_markdown()

        self.assertIn("Animal: 30_b4", markdown)
        self.assertIn("Dataset: B4", markdown)
        self.assertIn("Weight: 110.0 kg", markdown)
        self.assertIn("VALID", markdown)
        self.assertIn("Animal: 31_b4", markdown)
        self.assertIn("MISSING", markdown)

    def test_markdown_compact_animal_format(self) -> None:
        markdown = AnnotationQualityReport(self.report).to_markdown()

        # Animal 30_b4 has a valid SIDE image and no REAR image at all.
        self.assertIn("Animal: 30_b4", markdown)
        self.assertIn("SIDE", markdown)
        self.assertIn("REAR", markdown)
        self.assertIn("Image: 30_b4-1_s_110_F.jpg", markdown)
        self.assertIn("BBox COCO: [[10, 10, 100, 200]]", markdown)
        self.assertIn("Status: VALID", markdown)
        self.assertIn("Pairing: SIDE=VALID | REAR=MISSING", markdown)

        # Animal 31_b4 has a REAR image but with no annotation at all
        # (MISSING) and no SIDE image at all (also MISSING).
        self.assertIn("Animal: 31_b4", markdown)
        self.assertIn("Pairing: SIDE=MISSING | REAR=MISSING", markdown)

    def test_markdown_distinguishes_invalid_and_zero_bbox_from_missing(
        self,
    ) -> None:
        side_data = _coco(
            images=[_image(1, "40_b4-1_s_100_F.jpg")],
            annotations=[_annotation(1, 1, [0, 0, 0, 0])],
        )
        rear_data = _coco(
            images=[_image(1, "40_b4-1_r_100_F.jpg")],
            annotations=[_annotation(1, 1, [10, 10, -5, 50])],
        )

        report = self.auditor.audit([
            (side_data, DatasetType.B4, View.SIDE),
            (rear_data, DatasetType.B4, View.REAR),
        ])

        markdown = AnnotationQualityReport(report).to_markdown()

        # The image *exists* on both views; the bbox is what is wrong,
        # so the pairing line must not say "SIDE ONLY"/"REAR ONLY"/
        # "NO VALID VIEW" -- it must show the real per-view status.
        self.assertIn("Status: ZERO_BBOX", markdown)
        self.assertIn("Status: INVALID_BBOX", markdown)
        self.assertIn("Pairing: SIDE=ZERO_BBOX | REAR=INVALID_BBOX", markdown)
        self.assertNotIn("SIDE ONLY", markdown)
        self.assertNotIn("REAR ONLY", markdown)
        self.assertNotIn("NO VALID VIEW", markdown)

    def test_json_serialization(self) -> None:
        payload = AnnotationQualityReport(self.report).to_json_dict()

        # Must be JSON-serializable without custom encoders.
        text = json.dumps(payload)
        restored = json.loads(text)

        self.assertEqual(
            restored["summary"]["total_images"], 2
        )
        self.assertEqual(
            restored["summary"]["by_status"]["valid"], 1
        )
        self.assertEqual(
            restored["summary"]["by_status"]["missing"], 1
        )

    def test_csv_export(self, tmp_path: Path | None = None) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            csv_path = Path(tmp) / "audit.csv"

            AnnotationQualityReport(self.report).write_csv(csv_path)

            self.assertTrue(csv_path.exists())

            content = csv_path.read_text(encoding="utf-8")

            self.assertIn("30_b4-1_s_110_F.jpg", content)
            self.assertIn("valid", content)

    def test_audit_does_not_mutate_input(self) -> None:
        original_side = copy.deepcopy(self.side_data)
        original_rear = copy.deepcopy(self.rear_data)

        self.auditor.audit([
            (self.side_data, DatasetType.B4, View.SIDE),
            (self.rear_data, DatasetType.B4, View.REAR),
        ])

        self.assertEqual(self.side_data, original_side)
        self.assertEqual(self.rear_data, original_rear)

    def test_all_annotations_preserved_even_when_invalid(self) -> None:
        data = _coco(
            images=[_image(1, "40_b4-1_s_100_F.jpg")],
            annotations=[
                _annotation(1, 1, [0, 0, 0, 0]),
                _annotation(2, 1, [10, 10, 50, 50]),
            ],
        )

        report = self.auditor.audit_json(data, DatasetType.B4, View.SIDE)

        image = report.images[0]
        self.assertEqual(image.annotation_count, 2)
        statuses = {a.annotation_id: a.status for a in image.annotations}
        self.assertEqual(statuses[1], AnnotationStatus.ZERO_BBOX)
        self.assertEqual(statuses[2], AnnotationStatus.VALID)


if __name__ == "__main__":
    unittest.main()
