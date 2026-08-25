"""Independent, read-only audit for the project's COCO annotation files.

This module deliberately does not use :class:`COCOReader` and does not
receive a ``LivestockDataset``.  Its responsibility is to describe the
integrity and coverage of the source JSON files before they are attached to
the domain dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
from typing import Any

from src.core.context import ProjectContext


@dataclass(frozen=True, slots=True)
class COCOAuditTarget:
    """Location of one expected COCO export relative to ``Pixel``."""

    dataset: str
    folder: str
    filename: str

    def resolve(self, pixel_root: Path) -> Path:
        """Return the absolute annotation path for this target."""

        return pixel_root / self.dataset / self.folder / self.filename

    def resolve_images(self, pixel_root: Path) -> Path:
        """Return the physical image directory associated with this export."""

        return pixel_root / self.dataset / self.folder / "images"

    @property
    def label(self) -> str:
        """Human-readable identifier for reports and logs."""

        return f"{self.dataset}/{self.folder}"


@dataclass(slots=True)
class AnnotationCoverage:
    """Coverage and validity counters for one optional COCO field."""

    total_annotations: int = 0
    present: int = 0
    valid: int = 0
    invalid: int = 0

    @property
    def coverage_rate(self) -> float:
        """Percentage of annotations that provide this field."""

        if self.total_annotations == 0:
            return 0.0

        return self.present / self.total_annotations * 100.0


@dataclass(slots=True)
class PhysicalDatasetComparison:
    """Comparison between a COCO file and its physical image directory.

    Filename comparison is case-insensitive to match the behaviour of the
    Windows filesystem used by this project.  ``matching_rate`` is the
    Jaccard similarity of the declared and physical filename sets.
    """

    image_directory: Path
    image_directory_exists: bool
    disk_images: int = 0
    json_images: int = 0
    annotations: int = 0
    images_with_annotations: int = 0
    images_without_annotations: int = 0
    disk_only_images: int = 0
    json_only_images: int = 0
    matching_images: int = 0
    json_filenames_duplicate: int = 0
    image_ids_duplicate: int = 0
    orphan_annotation_image_ids: int = 0

    @property
    def matching_rate(self) -> float:
        """Exact filename overlap as a percentage of both source sets."""

        union = (
            self.matching_images
            + self.disk_only_images
            + self.json_only_images
        )
        if union == 0:
            return 0.0

        return self.matching_images / union * 100.0

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of this comparison."""

        result = asdict(self)
        result["image_directory"] = str(self.image_directory)
        result["matching_rate"] = self.matching_rate
        return result


@dataclass(slots=True)
class COCOAuditFileReport:
    """Read-only integrity report for one COCO JSON file."""

    target: str
    annotation_file: Path
    exists: bool
    images: int = 0
    annotations: int = 0
    categories: int = 0
    image_ids_duplicate: int = 0
    filenames_duplicate: int = 0
    category_ids_duplicate: int = 0
    orphan_annotation_image_ids: int = 0
    images_with_annotations: int = 0
    images_without_annotations: int = 0
    images_with_multiple_annotations: int = 0
    annotations_with_unknown_category: int = 0
    keypoint_distribution: dict[int, int] = field(default_factory=dict)
    invalid_keypoint_payloads: int = 0
    bbox: AnnotationCoverage = field(default_factory=AnnotationCoverage)
    segmentation: AnnotationCoverage = field(default_factory=AnnotationCoverage)
    segmentation_formats: dict[str, int] = field(default_factory=dict)
    structural_errors: list[str] = field(default_factory=list)
    physical_dataset: PhysicalDatasetComparison | None = None

    @property
    def image_annotation_match_rate(self) -> float:
        """Percentage of JSON images referenced by at least one annotation."""

        if self.images == 0:
            return 0.0

        return self.images_with_annotations / self.images * 100.0

    @property
    def is_readable(self) -> bool:
        """Whether the file exists and its top-level JSON could be read."""

        return self.exists and not self.structural_errors

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the report."""

        result = asdict(self)
        result["annotation_file"] = str(self.annotation_file)
        result["bbox"]["coverage_rate"] = self.bbox.coverage_rate
        result["segmentation"]["coverage_rate"] = self.segmentation.coverage_rate
        result["image_annotation_match_rate"] = self.image_annotation_match_rate
        if self.physical_dataset is not None:
            result["physical_dataset"] = self.physical_dataset.to_dict()
        return result


@dataclass(slots=True)
class COCOAuditReport:
    """Aggregate report for all configured COCO files."""

    files: list[COCOAuditFileReport] = field(default_factory=list)

    @property
    def files_found(self) -> int:
        """Number of configured JSON files available on disk."""

        return sum(item.exists for item in self.files)

    @property
    def total_images(self) -> int:
        """Total ``images`` entries across readable files."""

        return sum(item.images for item in self.files)

    @property
    def total_annotations(self) -> int:
        """Total ``annotations`` entries across readable files."""

        return sum(item.annotations for item in self.files)

    def to_dict(self) -> dict[str, Any]:
        """Return the complete report in a JSON-serializable form."""

        return {
            "files_found": self.files_found,
            "total_images": self.total_images,
            "total_annotations": self.total_annotations,
            "files": [item.to_dict() for item in self.files],
        }


class COCOAudit:
    """Audit COCO source files without enriching or changing the dataset.

    The target list is intentionally explicit.  It documents the six COCO
    exports currently in scope and prevents accidental inclusion of B3 or the
    duplicate files under ``Vector``.
    """

    _TARGETS: tuple[COCOAuditTarget, ...] = (
        COCOAuditTarget("B2", "Rear", "COCO_Rear.json"),
        COCOAuditTarget("B2", "Rear_2", "COCO_Rear_2.json"),
        COCOAuditTarget("B2", "Side", "COCO_Side.json"),
        COCOAuditTarget("B2", "Side_2", "COCO_Side_2.json"),
        COCOAuditTarget("B4", "Rear", "coco_b4_rear.json"),
        COCOAuditTarget("B4", "Side", "coco_b4_side.json"),
    )

    _REQUIRED_TOP_LEVEL_KEYS = ("images", "annotations", "categories")

    _IMAGE_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.config = context.config
        self.logger = context.logger

    def audit(self) -> COCOAuditReport:
        """Audit every configured COCO file and return its structured report."""

        self.logger.section("COCO Source Audit")

        report = COCOAuditReport(
            files=[self.audit_file(target) for target in self._TARGETS]
        )

        self.logger.info(f"COCO files found     : {report.files_found}")
        self.logger.info(f"COCO images          : {report.total_images}")
        self.logger.info(f"COCO annotations     : {report.total_annotations}")

        return report

    def audit_file(self, target: COCOAuditTarget) -> COCOAuditFileReport:
        """Audit one configured file, retaining errors in the returned report."""

        annotation_file = target.resolve(self.config.pixel_dataset)
        report = COCOAuditFileReport(
            target=target.label,
            annotation_file=annotation_file,
            exists=annotation_file.exists(),
        )

        if not report.exists:
            message = f"Annotation file not found: {annotation_file}"
            report.structural_errors.append(message)
            self.logger.warning(message)
            return report

        data = self._load_json(annotation_file, report)
        if data is None:
            return report

        collections = self._read_collections(data, report)
        if collections is None:
            return report

        images, annotations, categories = collections
        self._analyse(images, annotations, categories, report)
        report.physical_dataset = self._compare_physical_dataset(
            target,
            images,
            report,
        )
        self._log_file_summary(report)

        return report

    def _load_json(
        self,
        annotation_file: Path,
        report: COCOAuditFileReport,
    ) -> dict[str, Any] | None:
        try:
            with annotation_file.open("r", encoding="utf-8") as file:
                data = json.load(file)
        except (OSError, json.JSONDecodeError) as error:
            message = f"Unable to read {annotation_file}: {error}"
            report.structural_errors.append(message)
            self.logger.error(message)
            return None

        if not isinstance(data, dict):
            message = f"Top-level JSON value is not an object: {annotation_file}"
            report.structural_errors.append(message)
            self.logger.error(message)
            return None

        return data

    def _read_collections(
        self,
        data: dict[str, Any],
        report: COCOAuditFileReport,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]] | None:
        for key in self._REQUIRED_TOP_LEVEL_KEYS:
            if key not in data:
                report.structural_errors.append(f"Missing top-level key: {key}")

        if report.structural_errors:
            self.logger.error(
                f"{report.target}: invalid COCO structure: "
                + "; ".join(report.structural_errors)
            )
            return None

        values = tuple(data[key] for key in self._REQUIRED_TOP_LEVEL_KEYS)
        if not all(isinstance(value, list) for value in values):
            report.structural_errors.append(
                "Top-level images, annotations and categories must be arrays."
            )
            self.logger.error(f"{report.target}: invalid COCO collection types.")
            return None

        images, annotations, categories = values
        if not all(isinstance(item, dict) for item in images + annotations + categories):
            report.structural_errors.append(
                "COCO collections must contain JSON objects only."
            )
            self.logger.error(f"{report.target}: invalid item type in COCO collection.")
            return None

        return images, annotations, categories

    def _analyse(
        self,
        images: list[dict[str, Any]],
        annotations: list[dict[str, Any]],
        categories: list[dict[str, Any]],
        report: COCOAuditFileReport,
    ) -> None:
        report.images = len(images)
        report.annotations = len(annotations)
        report.categories = len(categories)
        report.bbox.total_annotations = report.annotations
        report.segmentation.total_annotations = report.annotations

        image_ids = [item.get("id") for item in images]
        filenames = [item.get("file_name") for item in images]
        category_ids = [item.get("id") for item in categories]
        report.image_ids_duplicate = self._duplicate_count(image_ids)
        report.filenames_duplicate = self._duplicate_count(filenames)
        report.category_ids_duplicate = self._duplicate_count(category_ids)

        valid_image_ids = set(image_ids)
        valid_category_ids = set(category_ids)
        annotations_by_image: Counter[Any] = Counter()

        for annotation in annotations:
            image_id = annotation.get("image_id")
            if image_id not in valid_image_ids:
                report.orphan_annotation_image_ids += 1
            else:
                annotations_by_image[image_id] += 1

            if annotation.get("category_id") not in valid_category_ids:
                report.annotations_with_unknown_category += 1

            self._analyse_keypoints(annotation, report)
            self._analyse_bbox(annotation, report.bbox)
            self._analyse_segmentation(annotation, report)

        report.images_with_annotations = sum(
            image_id in annotations_by_image for image_id in valid_image_ids
        )
        report.images_without_annotations = (
            report.images - report.images_with_annotations
        )
        report.images_with_multiple_annotations = sum(
            count > 1 for count in annotations_by_image.values()
        )

    def _compare_physical_dataset(
        self,
        target: COCOAuditTarget,
        images: list[dict[str, Any]],
        report: COCOAuditFileReport,
    ) -> PhysicalDatasetComparison:
        """Compare declared COCO filenames with supported image files on disk."""

        image_directory = target.resolve_images(self.config.pixel_dataset)
        comparison = PhysicalDatasetComparison(
            image_directory=image_directory,
            image_directory_exists=image_directory.is_dir(),
            json_images=report.images,
            annotations=report.annotations,
            images_with_annotations=report.images_with_annotations,
            images_without_annotations=report.images_without_annotations,
            json_filenames_duplicate=report.filenames_duplicate,
            image_ids_duplicate=report.image_ids_duplicate,
            orphan_annotation_image_ids=report.orphan_annotation_image_ids,
        )

        if not comparison.image_directory_exists:
            self.logger.warning(
                f"{target.label}: image directory not found: {image_directory}"
            )
            comparison.json_only_images = len(self._json_filename_set(images))
            return comparison

        disk_filenames = {
            image.name.casefold()
            for image in image_directory.iterdir()
            if image.is_file() and image.suffix.casefold() in self._IMAGE_EXTENSIONS
        }
        json_filenames = self._json_filename_set(images)

        comparison.disk_images = len(disk_filenames)
        comparison.matching_images = len(disk_filenames & json_filenames)
        comparison.disk_only_images = len(disk_filenames - json_filenames)
        comparison.json_only_images = len(json_filenames - disk_filenames)

        return comparison

    @staticmethod
    def _json_filename_set(images: list[dict[str, Any]]) -> set[str]:
        """Return comparable COCO filenames, excluding absent/invalid values."""

        return {
            filename.casefold()
            for image in images
            if isinstance((filename := image.get("file_name")), str)
        }

    @staticmethod
    def _duplicate_count(values: list[Any]) -> int:
        """Count distinct duplicated values, ignoring missing identifiers."""

        counts = Counter(value for value in values if value is not None)
        return sum(count > 1 for count in counts.values())

    @staticmethod
    def _analyse_keypoints(
        annotation: dict[str, Any],
        report: COCOAuditFileReport,
    ) -> None:
        keypoints = annotation.get("keypoints", [])
        if not isinstance(keypoints, list) or len(keypoints) % 3 != 0:
            report.invalid_keypoint_payloads += 1
            return

        count = len(keypoints) // 3
        report.keypoint_distribution[count] = (
            report.keypoint_distribution.get(count, 0) + 1
        )

    @staticmethod
    def _analyse_bbox(
        annotation: dict[str, Any],
        coverage: AnnotationCoverage,
    ) -> None:
        bbox = annotation.get("bbox")
        if bbox is None:
            return

        coverage.present += 1
        if (
            isinstance(bbox, list)
            and len(bbox) == 4
            and all(isinstance(value, (int, float)) for value in bbox)
            and bbox[0] >= 0
            and bbox[1] >= 0
            and bbox[2] > 0
            and bbox[3] > 0
        ):
            coverage.valid += 1
        else:
            coverage.invalid += 1

    def _analyse_segmentation(
        self,
        annotation: dict[str, Any],
        report: COCOAuditFileReport,
    ) -> None:
        segmentation = annotation.get("segmentation")
        coverage = report.segmentation

        if segmentation is None or segmentation == []:
            self._increase_format(report, "empty")
            return

        coverage.present += 1
        if isinstance(segmentation, dict):
            self._increase_format(report, "rle")
            if "counts" in segmentation and "size" in segmentation:
                coverage.valid += 1
            else:
                coverage.invalid += 1
            return

        if isinstance(segmentation, list):
            self._increase_format(report, "polygon")
            if self._valid_polygons(segmentation):
                coverage.valid += 1
            else:
                coverage.invalid += 1
            return

        self._increase_format(report, "unsupported")
        coverage.invalid += 1

    @staticmethod
    def _valid_polygons(segmentation: list[Any]) -> bool:
        """Validate COCO polygon structure without rasterising its geometry."""

        return bool(segmentation) and all(
            isinstance(polygon, list)
            and len(polygon) >= 6
            and len(polygon) % 2 == 0
            and all(isinstance(value, (int, float)) for value in polygon)
            for polygon in segmentation
        )

    @staticmethod
    def _increase_format(report: COCOAuditFileReport, name: str) -> None:
        report.segmentation_formats[name] = (
            report.segmentation_formats.get(name, 0) + 1
        )

    def _log_file_summary(self, report: COCOAuditFileReport) -> None:
        comparison = report.physical_dataset
        matching_rate = (
            comparison.matching_rate
            if comparison is not None
            else 0.0
        )
        self.logger.info(
            f"{report.target}: images={report.images}, "
            f"annotations={report.annotations}, "
            f"categories={report.categories}, "
            f"orphan_image_ids={report.orphan_annotation_image_ids}, "
            f"annotation_rate={report.image_annotation_match_rate:.2f}%, "
            f"physical_match_rate={matching_rate:.2f}%"
        )
