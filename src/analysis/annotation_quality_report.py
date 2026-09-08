"""
Annotation quality audit report generation.

Turns an :class:`~src.analysis.annotation_quality_audit.AnnotationQualityAuditReport`
into a human-readable Markdown/TXT report for the thesis advisor, plus
machine-readable JSON/CSV exports for a later cleaning stage.

This module never modifies the underlying COCO JSON files; it only reads
an already-computed audit report and writes new output files.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from src.analysis.annotation_quality_audit import (
    AnnotationQualityAuditReport,
    ImageQualityAuditRecord,
)
from src.coco.enums import AnnotationStatus


# Priority order (highest first) used to summarize the overall status of
# a view (SIDE/REAR) when it holds more than one image/annotation. This
# never changes the underlying per-image/per-annotation classification;
# it only picks the single most informative label to display.
_STATUS_PRIORITY: list[AnnotationStatus] = [

    AnnotationStatus.VALID,

    AnnotationStatus.MULTIPLE_ANNOTATIONS,

    AnnotationStatus.OUT_OF_BOUNDS,

    AnnotationStatus.INVALID_BBOX,

    AnnotationStatus.ZERO_BBOX,

    AnnotationStatus.MISSING,

]


class AnnotationQualityReport:
    """
    Generates human-readable and machine-readable reports from an
    :class:`AnnotationQualityAuditReport`.
    """

    def __init__(self, report: AnnotationQualityAuditReport) -> None:

        self.report = report

    # =====================================================
    # Human-readable Markdown/TXT report
    # =====================================================

    def to_markdown(self) -> str:
        """
        Build the full advisor-facing Markdown report as a string.
        """

        lines: list[str] = []

        summary = self.report.summary

        lines.append("# Annotation Quality Audit Report")
        lines.append("")

        # -------------------------------------------------
        # Overall summary
        # -------------------------------------------------

        lines.append("## Overall Summary")
        lines.append("")
        lines.append(f"- Total images: {summary.total_images}")
        lines.append(f"- Total annotations: {summary.total_annotations}")
        lines.append(f"- Total animals: {summary.total_animals}")
        lines.append("")

        # -------------------------------------------------
        # By dataset / view / status
        # -------------------------------------------------

        lines.append("## By Dataset")
        lines.append("")
        for key, value in sorted(summary.by_dataset.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("## By View")
        lines.append("")
        for key, value in sorted(summary.by_view.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

        lines.append("## By Annotation Status")
        lines.append("")
        for key, value in sorted(summary.by_status.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

        # -------------------------------------------------
        # Animal pairing statistics
        # -------------------------------------------------

        lines.append("## Animal Pairing Statistics")
        lines.append("")
        for key, value in sorted(summary.by_pairing.items()):
            lines.append(f"- {key}: {value}")
        lines.append("")

        # -------------------------------------------------
        # Detailed animal-by-animal records
        # -------------------------------------------------

        lines.append("## Animal-by-Animal Detail")
        lines.append("")

        for animal in self.report.sorted_animals:

            lines.append(f"Animal: {animal.animal_id}")

            if animal.dataset is not None:
                lines.append(f"Dataset: {animal.dataset.value}")

            if animal.weight_kg is not None:
                lines.append(f"Weight: {animal.weight_kg} kg")

            if animal.sex is not None:
                lines.append(f"Sex: {animal.sex.value}")

            lines.append("SIDE")
            lines.extend(self._render_view_lines(animal.side_images))

            lines.append("REAR")
            lines.extend(self._render_view_lines(animal.rear_images))

            side_status = self._view_status_label(animal.side_images)
            rear_status = self._view_status_label(animal.rear_images)

            lines.append(
                f"Pairing: SIDE={side_status} | REAR={rear_status}"
            )
            lines.append("")

        # -------------------------------------------------
        # Unparsed / unpaired images
        # -------------------------------------------------

        unparsed = [
            image
            for image in self.report.sorted_images
            if not image.animal_id
        ]

        if unparsed:

            lines.append("## Images Without Parsed Animal Identity")
            lines.append("")

            for image in unparsed:

                lines.append(
                    f"- {image.file_name} "
                    f"(status={image.status.value}, "
                    f"parse_error={image.parse_error})"
                )

            lines.append("")

        return "\n".join(lines)

    @staticmethod
    def _render_view_lines(
        images: list[ImageQualityAuditRecord],
    ) -> list[str]:
        """
        Render one line per image inside a SIDE/REAR block: image name,
        original COCO bbox, and image-level status. A missing view (no
        image at all) is rendered as a single explicit MISSING line.
        """

        if not images:

            return ["  Status: MISSING"]

        lines: list[str] = []

        for image in images:

            lines.append(f"  Image: {image.file_name}")

            bbox_values = [
                annotation.bbox_xywh
                for annotation in image.annotations
                if annotation.bbox_xywh is not None
            ]

            if bbox_values:

                lines.append(
                    f"  BBox COCO: {[list(value) for value in bbox_values]}"
                )

            lines.append(f"  Status: {image.status.value.upper()}")

        return lines

    @staticmethod
    def _view_status_label(
        images: list[ImageQualityAuditRecord],
    ) -> str:
        """
        Return a single, real status label summarizing one animal's view
        (SIDE or REAR), distinguishing VALID / INVALID_BBOX / ZERO_BBOX /
        OUT_OF_BOUNDS / MULTIPLE_ANNOTATIONS / MISSING instead of
        collapsing everything into "present/absent".
        """

        if not images:

            return AnnotationStatus.MISSING.value.upper()

        statuses = {image.status for image in images}

        for candidate in _STATUS_PRIORITY:

            if candidate in statuses:

                return candidate.value.upper()

        # Defensive fallback; every AnnotationStatus is covered above.
        return next(iter(statuses)).value.upper()

    def write_markdown(self, path: Path) -> None:
        """
        Write the human-readable report to ``path``.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        path.write_text(self.to_markdown(), encoding="utf-8")

    # =====================================================
    # Machine-readable JSON
    # =====================================================

    def to_json_dict(self) -> dict[str, Any]:
        """
        Return the complete audit result as a JSON-serializable dict.
        """

        return self.report.to_dict()

    def write_json(self, path: Path) -> None:
        """
        Write the complete audit result as JSON.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as file:

            json.dump(self.to_json_dict(), file, indent=4)

    # =====================================================
    # Machine-readable CSV (per-image records)
    # =====================================================

    _CSV_FIELDS = [
        "dataset",
        "animal_id",
        "view",
        "image_id",
        "file_name",
        "width",
        "height",
        "weight_kg",
        "sex",
        "annotation_count",
        "annotation_ids",
        "status",
        "bbox_xywh",
        "bbox_xyxy",
        "issues",
    ]

    def to_csv_rows(self) -> list[dict[str, Any]]:
        """
        Return one row per image, in deterministic order
        (dataset, animal_id, view, file_name).
        """

        rows: list[dict[str, Any]] = []

        for image in self.report.sorted_images:

            bbox_xywh = [
                annotation.bbox_xywh
                for annotation in image.annotations
                if annotation.bbox_xywh is not None
            ]

            bbox_xyxy = [
                annotation.bbox_xyxy
                for annotation in image.annotations
                if annotation.bbox_xyxy is not None
            ]

            rows.append({

                "dataset": image.dataset.value if image.dataset else "",

                "animal_id": image.animal_id or "",

                "view": image.view.value if image.view else "",

                "image_id": image.image_id,

                "file_name": image.file_name,

                "width": image.width,

                "height": image.height,

                "weight_kg": image.weight_kg,

                "sex": image.sex.value if image.sex else "",

                "annotation_count": image.annotation_count,

                "annotation_ids": image.annotation_ids,

                "status": image.status.value,

                "bbox_xywh": bbox_xywh,

                "bbox_xyxy": bbox_xyxy,

                "issues": "; ".join(image.issues),

            })

        return rows

    def write_csv(self, path: Path) -> None:
        """
        Write the per-image audit records to CSV.
        """

        path.parent.mkdir(parents=True, exist_ok=True)

        rows = self.to_csv_rows()

        with path.open("w", encoding="utf-8", newline="") as file:

            writer = csv.DictWriter(file, fieldnames=self._CSV_FIELDS)

            writer.writeheader()

            for row in rows:

                writer.writerow(row)
