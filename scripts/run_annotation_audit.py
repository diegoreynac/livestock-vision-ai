"""
Run the annotation quality audit against the project's COCO JSON exports.

This script is a thin orchestration layer: it loads the four existing COCO
JSON files, feeds them into the existing ``AnnotationQualityAuditor``
(one audit per dataset, combining its Side and Rear exports), and writes
the existing ``AnnotationQualityReport`` outputs (Markdown/JSON/CSV) to
``output/annotation_audit/<DATASET>/``.

It does not implement any auditing logic itself and does not modify the
source COCO JSON files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

_PROJECT_ROOT_FOR_IMPORTS = Path(__file__).resolve().parents[1]

if str(_PROJECT_ROOT_FOR_IMPORTS) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT_FOR_IMPORTS))

from src.analysis.annotation_quality_audit import (
    AnnotationQualityAuditor,
    AnnotationQualityAuditReport,
)
from src.analysis.annotation_quality_report import AnnotationQualityReport
from src.dataset.enums import DatasetType, View

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

OUTPUT_ROOT = PROJECT_ROOT / "output" / "annotation_audit"

# Each dataset combines its Side and Rear COCO exports into a single audit.
DATASET_SOURCES: dict[DatasetType, list[tuple[Path, View]]] = {

    DatasetType.B2: [
        (PROJECT_ROOT / "COCO_Side.json", View.SIDE),
        (PROJECT_ROOT / "COCO_Rear.json", View.REAR),
    ],

    DatasetType.B4: [
        (PROJECT_ROOT / "coco_b4_side.json", View.SIDE),
        (PROJECT_ROOT / "coco_b4_rear.json", View.REAR),
    ],

}


# ==========================================================
# Public API
# ==========================================================

def run_audit_for_dataset(
    dataset: DatasetType,
    sources: list[tuple[Path, View]],
    auditor: AnnotationQualityAuditor | None = None,
) -> AnnotationQualityAuditReport:
    """
    Audit one dataset's COCO JSON exports (Side + Rear) and return the
    resulting :class:`AnnotationQualityAuditReport`.

    Reads each COCO JSON file without modifying it.
    """

    auditor = auditor or AnnotationQualityAuditor()

    loaded: list[tuple[dict, DatasetType, View]] = []

    for annotation_file, view in sources:

        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        loaded.append((data, dataset, view))

    return auditor.audit(loaded)


def write_reports(
    dataset: DatasetType,
    report: AnnotationQualityAuditReport,
    output_root: Path = OUTPUT_ROOT,
) -> dict[str, Path]:
    """
    Write the Markdown/JSON/CSV outputs for one dataset's audit report.

    Returns the paths that were written.
    """

    dataset_dir = output_root / dataset.value

    generator = AnnotationQualityReport(report)

    markdown_path = dataset_dir / "annotation_audit.md"
    json_path = dataset_dir / "annotation_audit.json"
    csv_path = dataset_dir / "annotation_audit.csv"

    generator.write_markdown(markdown_path)
    generator.write_json(json_path)
    generator.write_csv(csv_path)

    return {
        "markdown": markdown_path,
        "json": json_path,
        "csv": csv_path,
    }


def print_summary(
    dataset: DatasetType,
    report: AnnotationQualityAuditReport,
) -> None:
    """
    Print a terminal summary for one dataset: total images, annotations,
    and BBox status breakdown.
    """

    summary = report.summary

    print(f"\n{dataset.value}")
    print("-" * 40)
    print(f"Total images      : {summary.total_images}")
    print(f"Total annotations  : {summary.total_annotations}")
    print(f"Total animals      : {summary.total_animals}")
    print("BBox status:")

    for status, count in sorted(summary.by_status.items()):
        print(f"  {status:<20}: {count}")

    print("Animal pairing:")

    for pairing, count in sorted(summary.by_pairing.items()):
        print(f"  {pairing:<20}: {count}")


# ==========================================================
# Entry point
# ==========================================================

def main() -> None:

    auditor = AnnotationQualityAuditor()

    for dataset, sources in DATASET_SOURCES.items():

        report = run_audit_for_dataset(dataset, sources, auditor)

        write_reports(dataset, report)

        print_summary(dataset, report)


if __name__ == "__main__":

    main()
