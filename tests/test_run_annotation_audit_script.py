"""
Tests for the ``scripts/run_annotation_audit.py`` runner.

These tests do not use the real project COCO JSON files or the real
``output/`` directory: they build small synthetic COCO structures and
write to a temporary directory instead.
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from scripts.run_annotation_audit import (
    print_summary,
    run_audit_for_dataset,
    write_reports,
)
from src.dataset.enums import DatasetType, View


def _coco(images: list[dict], annotations: list[dict]) -> dict:

    return {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": 1, "name": "cow"}],
    }


class TestRunAnnotationAuditScript(unittest.TestCase):

    def setUp(self) -> None:

        self.tmp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp_dir.cleanup)

        self.tmp_path = Path(self.tmp_dir.name)

        self.side_file = self.tmp_path / "side.json"
        self.rear_file = self.tmp_path / "rear.json"

        side_data = _coco(
            images=[
                {
                    "id": 1,
                    "file_name": "30_b4-1_s_110_F.jpg",
                    "width": 1920,
                    "height": 1080,
                }
            ],
            annotations=[
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [10, 10, 100, 200],
                }
            ],
        )

        rear_data = _coco(images=[], annotations=[])

        self.side_file.write_text(json.dumps(side_data), encoding="utf-8")
        self.rear_file.write_text(json.dumps(rear_data), encoding="utf-8")

    def test_run_audit_for_dataset_reads_files_and_produces_report(self) -> None:

        report = run_audit_for_dataset(
            DatasetType.B4,
            [
                (self.side_file, View.SIDE),
                (self.rear_file, View.REAR),
            ],
        )

        self.assertEqual(report.summary.total_images, 1)
        self.assertEqual(report.summary.total_annotations, 1)
        self.assertEqual(report.summary.by_status["valid"], 1)

    def test_write_reports_creates_expected_output_paths(self) -> None:

        report = run_audit_for_dataset(
            DatasetType.B4,
            [
                (self.side_file, View.SIDE),
                (self.rear_file, View.REAR),
            ],
        )

        output_root = self.tmp_path / "output" / "annotation_audit"

        paths = write_reports(DatasetType.B4, report, output_root=output_root)

        expected_dir = output_root / "B4"

        self.assertEqual(
            paths["markdown"], expected_dir / "annotation_audit.md"
        )
        self.assertEqual(
            paths["json"], expected_dir / "annotation_audit.json"
        )
        self.assertEqual(
            paths["csv"], expected_dir / "annotation_audit.csv"
        )

        for path in paths.values():
            self.assertTrue(path.exists())

        # JSON output must be loadable and consistent with the report.
        payload = json.loads(paths["json"].read_text(encoding="utf-8"))
        self.assertEqual(payload["summary"]["total_images"], 1)

    def test_print_summary_reports_totals_and_bbox_statuses(self) -> None:

        report = run_audit_for_dataset(
            DatasetType.B4,
            [
                (self.side_file, View.SIDE),
                (self.rear_file, View.REAR),
            ],
        )

        import io
        from contextlib import redirect_stdout

        buffer = io.StringIO()

        with redirect_stdout(buffer):
            print_summary(DatasetType.B4, report)

        output = buffer.getvalue()

        self.assertIn("B4", output)
        self.assertIn("Total images", output)
        self.assertIn("Total annotations", output)
        self.assertIn("valid", output)

    def test_does_not_modify_source_json_files(self) -> None:

        original_side = self.side_file.read_text(encoding="utf-8")
        original_rear = self.rear_file.read_text(encoding="utf-8")

        run_audit_for_dataset(
            DatasetType.B4,
            [
                (self.side_file, View.SIDE),
                (self.rear_file, View.REAR),
            ],
        )

        self.assertEqual(self.side_file.read_text(encoding="utf-8"), original_side)
        self.assertEqual(self.rear_file.read_text(encoding="utf-8"), original_rear)


if __name__ == "__main__":
    unittest.main()
