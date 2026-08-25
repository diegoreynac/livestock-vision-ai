"""Validation and coverage reporting for annotations attached to a dataset."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

from src.coco.enums import KeypointVisibility
from src.coco.models import COCOAnnotation
from src.coco.validation_statistics import COCOValidationStatistics
from src.core.context import ProjectContext
from src.dataset.livestock_dataset import LivestockDataset
from src.dataset.models import ImageRecord


@dataclass(slots=True)
class AnnotationFieldCoverage:
    """Availability, validity and absence counters for one annotation field."""

    available: int = 0
    missing: int = 0
    valid: int = 0
    invalid: int = 0

    def reset(self) -> None:
        """Reset all counters for a new validation run."""

        self.available = 0
        self.missing = 0
        self.valid = 0
        self.invalid = 0


@dataclass(slots=True)
class COCOValidationCoverage:
    """Coverage report that distinguishes missing optional data from defects."""

    annotations_available: int = 0
    annotations_missing: int = 0
    bbox: AnnotationFieldCoverage = field(default_factory=AnnotationFieldCoverage)
    segmentation: AnnotationFieldCoverage = field(
        default_factory=AnnotationFieldCoverage
    )
    keypoints: AnnotationFieldCoverage = field(
        default_factory=AnnotationFieldCoverage
    )

    def reset(self) -> None:
        """Reset all coverage counters for a new validation run."""

        self.annotations_available = 0
        self.annotations_missing = 0
        self.bbox.reset()
        self.segmentation.reset()
        self.keypoints.reset()

    def __str__(self) -> str:
        """Format coverage as a concise report for the project logger."""

        return (
            "\n"
            "COCO Validation Coverage\n"
            "------------------------\n"
            f"Annotations available : {self.annotations_available}\n"
            f"Annotations missing   : {self.annotations_missing}\n"
            "\n"
            "BBox\n"
            f"  Available : {self.bbox.available}\n"
            f"  Missing   : {self.bbox.missing}\n"
            f"  Valid     : {self.bbox.valid}\n"
            f"  Invalid   : {self.bbox.invalid}\n"
            "Segmentation\n"
            f"  Available : {self.segmentation.available}\n"
            f"  Missing   : {self.segmentation.missing}\n"
            f"  Valid     : {self.segmentation.valid}\n"
            f"  Invalid   : {self.segmentation.invalid}\n"
            "Keypoints\n"
            f"  Available : {self.keypoints.available}\n"
            f"  Missing   : {self.keypoints.missing}\n"
            f"  Valid     : {self.keypoints.valid}\n"
            f"  Invalid   : {self.keypoints.invalid}"
        )


class COCOValidator:
    """Validate available COCO fields and report their independent coverage."""

    def __init__(self, context: ProjectContext) -> None:
        self.context = context
        self.config = context.config
        self.logger = context.logger
        self.statistics = COCOValidationStatistics()
        self.coverage = COCOValidationCoverage()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def validate(self, dataset: LivestockDataset) -> COCOValidationStatistics:
        """Validate the attached annotations without requiring optional fields."""

        self.statistics.reset()
        self.coverage.reset()
        self.logger.section("COCO Validation")

        for record in dataset:
            self.statistics.images_checked += 1
            self._validate_annotation(record)

        self.logger.info(str(self.statistics))
        self.logger.info(str(self.coverage))
        return self.statistics

    # --------------------------------------------------
    # Annotation validation
    # --------------------------------------------------

    def _validate_annotation(self, record: ImageRecord) -> None:
        annotation = record.annotation
        if annotation is None:
            self.statistics.missing_annotations += 1
            self.coverage.annotations_missing += 1
            self._warning(f"{record.filename}: Missing annotation.")
            return

        self.statistics.annotations_checked += 1
        self.coverage.annotations_available += 1
        self._validate_bbox(record, annotation)
        self._validate_segmentation(record, annotation)
        self._validate_keypoints(record, annotation)

    def _validate_bbox(
        self,
        record: ImageRecord,
        annotation: COCOAnnotation,
    ) -> None:
        """Classify bbox as missing, valid or invalid without false errors."""

        bbox = getattr(annotation, "bbox", None)
        coverage = self.coverage.bbox
        if bbox is None or self._is_empty_bbox(bbox):
            coverage.missing += 1
            return

        coverage.available += 1
        if self._is_valid_bbox(bbox):
            coverage.valid += 1
            return

        coverage.invalid += 1
        self.statistics.bbox_errors += 1
        self._error(f"{record.filename}: Invalid bounding box.")

    def _validate_segmentation(
        self,
        record: ImageRecord,
        annotation: COCOAnnotation,
    ) -> None:
        """Classify optional polygon or RLE segmentation coverage."""

        segmentation = getattr(annotation, "segmentation", None)
        coverage = self.coverage.segmentation
        if segmentation is None or segmentation == []:
            coverage.missing += 1
            return

        coverage.available += 1
        if self._is_valid_segmentation(segmentation):
            coverage.valid += 1
            return

        coverage.invalid += 1
        self.statistics.segmentation_errors += 1
        self._error(f"{record.filename}: Invalid segmentation.")

    def _validate_keypoints(
        self,
        record: ImageRecord,
        annotation: COCOAnnotation,
    ) -> None:
        """Classify optional keypoints independently of their cardinality."""

        keypoints = getattr(annotation, "keypoints", None)
        coverage = self.coverage.keypoints
        if keypoints is None or keypoints == []:
            coverage.missing += 1
            return

        coverage.available += 1
        if self._are_valid_keypoints(keypoints):
            coverage.valid += 1
            return

        coverage.invalid += 1
        self.statistics.keypoint_errors += 1
        self._error(f"{record.filename}: Invalid keypoints.")

    # --------------------------------------------------
    # Field predicates
    # --------------------------------------------------

    @classmethod
    def _is_empty_bbox(cls, bbox: object) -> bool:
        """Recognise the ``[0, 0, 0, 0]`` placeholder used by B4 exports."""

        values = (
            getattr(bbox, "x", None),
            getattr(bbox, "y", None),
            getattr(bbox, "width", None),
            getattr(bbox, "height", None),
        )
        return all(value == 0 for value in values)

    @classmethod
    def _is_valid_bbox(cls, bbox: object) -> bool:
        values = (
            getattr(bbox, "x", None),
            getattr(bbox, "y", None),
            getattr(bbox, "width", None),
            getattr(bbox, "height", None),
        )
        if not cls._finite_numbers(*values):
            return False

        x, y, width, height = values
        return x >= 0 and y >= 0 and width > 0 and height > 0

    @classmethod
    def _is_valid_segmentation(cls, segmentation: object) -> bool:
        if isinstance(segmentation, dict):
            return "counts" in segmentation and "size" in segmentation

        if not isinstance(segmentation, list) or not segmentation:
            return False

        return all(
            isinstance(polygon, list)
            and len(polygon) >= 6
            and len(polygon) % 2 == 0
            and cls._finite_numbers(*polygon)
            for polygon in segmentation
        )

    @classmethod
    def _are_valid_keypoints(cls, keypoints: object) -> bool:
        if not isinstance(keypoints, list):
            return False

        for keypoint in keypoints:
            x = getattr(keypoint, "x", None)
            y = getattr(keypoint, "y", None)
            visibility = getattr(keypoint, "visibility", None)
            if not cls._finite_numbers(x, y) or x < 0 or y < 0:
                return False
            if not isinstance(visibility, KeypointVisibility):
                return False

        return True

    @staticmethod
    def _finite_numbers(*values: object) -> bool:
        return all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _warning(self, message: str) -> None:
        self.statistics.warnings += 1
        self.logger.warning(message)

    def _error(self, message: str) -> None:
        self.statistics.errors += 1
        self.logger.error(message)
