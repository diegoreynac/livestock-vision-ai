"""
COCO annotation quality audit.

Read-only, per-image and per-animal audit of COCO annotation JSON files.

This module answers exactly one question: "what is the quality of the
annotations already present in the COCO exports?". It never repairs,
clips, normalizes, or removes anything -- it only detects, classifies,
counts, groups and reports.

The audit operates purely on COCO JSON metadata (``images`` / ``annotations``
lists). The underlying image files are never required or opened.

Filename parsing (dataset/view/animal identity) reuses the project's
existing :class:`src.dataset.parser.FilenameParser` instead of duplicating
that logic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.coco.enums import AnnotationStatus
from src.dataset.enums import DatasetType, Sex, View
from src.dataset.models import ImageFolder, ImageRecord
from src.dataset.parser import FilenameParser


# ==========================================================
# Animal-level pairing status
# ==========================================================

class AnimalPairingStatus(Enum):
    """
    Side/Rear pairing outcome for a single animal.
    """

    SIDE_REAR_VALID = "side_rear_valid"

    SIDE_ONLY = "side_only"

    REAR_ONLY = "rear_only"

    NO_VALID_VIEW = "no_valid_view"

    def __str__(self) -> str:

        return self.value


# ==========================================================
# Per-annotation audit record
# ==========================================================

@dataclass(slots=True)
class BBoxAuditRecord:
    """
    Read-only audit of a single COCO annotation's bounding box.
    """

    annotation_id: int | None

    bbox_xywh: tuple[Any, Any, Any, Any] | None

    bbox_xyxy: tuple[float, float, float, float] | None

    status: AnnotationStatus

    issues: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:

        return {

            "annotation_id": self.annotation_id,

            "bbox_xywh": list(self.bbox_xywh) if self.bbox_xywh is not None else None,

            "bbox_xyxy": list(self.bbox_xyxy) if self.bbox_xyxy is not None else None,

            "status": self.status.value,

            "issues": list(self.issues),

        }


# ==========================================================
# Per-image audit record
# ==========================================================

@dataclass(slots=True)
class ImageQualityAuditRecord:
    """
    Read-only audit of a single COCO image entry and every annotation
    that references it.
    """

    image_id: int | None

    file_name: str | None

    width: int | None

    height: int | None

    dataset: DatasetType | None = None

    view: View | None = None

    animal_id: str | None = None

    weight_kg: float | None = None

    sex: Sex | None = None

    parse_error: str | None = None

    annotations: list[BBoxAuditRecord] = field(default_factory=list)

    status: AnnotationStatus = AnnotationStatus.MISSING

    issues: list[str] = field(default_factory=list)

    @property
    def annotation_count(self) -> int:

        return len(self.annotations)

    @property
    def annotation_ids(self) -> list[int | None]:

        return [item.annotation_id for item in self.annotations]

    @property
    def has_valid_annotation(self) -> bool:
        """
        True if at least one annotation on this image is VALID, regardless
        of the image-level overall status (which may be
        ``MULTIPLE_ANNOTATIONS`` even when a valid annotation exists).
        """

        return any(
            item.status is AnnotationStatus.VALID
            for item in self.annotations
        )

    def to_dict(self) -> dict[str, Any]:

        return {

            "image_id": self.image_id,

            "file_name": self.file_name,

            "width": self.width,

            "height": self.height,

            "dataset": self.dataset.value if self.dataset else None,

            "view": self.view.value if self.view else None,

            "animal_id": self.animal_id,

            "weight_kg": self.weight_kg,

            "sex": self.sex.value if self.sex else None,

            "parse_error": self.parse_error,

            "annotation_count": self.annotation_count,

            "annotation_ids": self.annotation_ids,

            "status": self.status.value,

            "issues": list(self.issues),

            "annotations": [item.to_dict() for item in self.annotations],

        }


# ==========================================================
# Per-animal audit record
# ==========================================================

@dataclass(slots=True)
class AnimalQualityAuditRecord:
    """
    Groups per-image audit records by animal identity and relates
    Side/Rear views for that animal.
    """

    animal_id: str

    dataset: DatasetType | None

    weight_kg: float | None

    sex: Sex | None

    side_images: list[ImageQualityAuditRecord] = field(default_factory=list)

    rear_images: list[ImageQualityAuditRecord] = field(default_factory=list)

    pairing_status: AnimalPairingStatus = AnimalPairingStatus.NO_VALID_VIEW

    @property
    def has_valid_side(self) -> bool:

        return any(image.has_valid_annotation for image in self.side_images)

    @property
    def has_valid_rear(self) -> bool:

        return any(image.has_valid_annotation for image in self.rear_images)

    def to_dict(self) -> dict[str, Any]:

        return {

            "animal_id": self.animal_id,

            "dataset": self.dataset.value if self.dataset else None,

            "weight_kg": self.weight_kg,

            "sex": self.sex.value if self.sex else None,

            "pairing_status": self.pairing_status.value,

            "side_images": [image.to_dict() for image in self.side_images],

            "rear_images": [image.to_dict() for image in self.rear_images],

        }


# ==========================================================
# Aggregate summary
# ==========================================================

@dataclass(slots=True)
class AnnotationQualitySummary:
    """
    Deterministic, JSON-serializable aggregate statistics.
    """

    total_images: int = 0

    total_annotations: int = 0

    total_animals: int = 0

    by_dataset: dict[str, int] = field(default_factory=dict)

    by_view: dict[str, int] = field(default_factory=dict)

    by_status: dict[str, int] = field(default_factory=dict)

    by_pairing: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:

        return {

            "total_images": self.total_images,

            "total_annotations": self.total_annotations,

            "total_animals": self.total_animals,

            "by_dataset": dict(sorted(self.by_dataset.items())),

            "by_view": dict(sorted(self.by_view.items())),

            "by_status": dict(sorted(self.by_status.items())),

            "by_pairing": dict(sorted(self.by_pairing.items())),

        }


# ==========================================================
# Full report
# ==========================================================

@dataclass(slots=True)
class AnnotationQualityAuditReport:
    """
    Complete, read-only annotation quality audit result.
    """

    images: list[ImageQualityAuditRecord] = field(default_factory=list)

    animals: list[AnimalQualityAuditRecord] = field(default_factory=list)

    summary: AnnotationQualitySummary = field(
        default_factory=AnnotationQualitySummary
    )

    @property
    def sorted_images(self) -> list[ImageQualityAuditRecord]:
        """
        Deterministic ordering: dataset, animal_id, view, file_name.
        """

        def sort_key(image: ImageQualityAuditRecord):

            return (
                image.dataset.value if image.dataset else "",
                image.animal_id or "",
                image.view.value if image.view else "",
                image.file_name or "",
            )

        return sorted(self.images, key=sort_key)

    @property
    def sorted_animals(self) -> list[AnimalQualityAuditRecord]:
        """
        Deterministic ordering: dataset, animal_id.
        """

        def sort_key(animal: AnimalQualityAuditRecord):

            return (
                animal.dataset.value if animal.dataset else "",
                animal.animal_id,
            )

        return sorted(self.animals, key=sort_key)

    def to_dict(self) -> dict[str, Any]:

        return {

            "summary": self.summary.to_dict(),

            "animals": [
                animal.to_dict() for animal in self.sorted_animals
            ],

            "images": [
                image.to_dict() for image in self.sorted_images
            ],

        }


# ==========================================================
# Auditor
# ==========================================================

class AnnotationQualityAuditor:
    """
    Builds an :class:`AnnotationQualityAuditReport` from one or more COCO
    JSON structures.

    This class never mutates the input COCO structures and never touches
    image files on disk.
    """

    def __init__(self) -> None:

        self._parser = FilenameParser()

    # =====================================================
    # Public API
    # =====================================================

    def audit(
        self,
        sources: list[tuple[dict[str, Any], DatasetType, View]],
    ) -> AnnotationQualityAuditReport:
        """
        Audit multiple COCO JSON structures (already parsed into ``dict``
        objects), each tagged with the dataset/view it belongs to.
        """

        images: list[ImageQualityAuditRecord] = []

        for data, dataset, view in sources:

            images.extend(
                self._audit_single(data, dataset, view)
            )

        animals = self._group_by_animal(images)

        summary = self._summarize(images, animals)

        return AnnotationQualityAuditReport(
            images=images,
            animals=animals,
            summary=summary,
        )

    def audit_json(
        self,
        data: dict[str, Any],
        dataset: DatasetType,
        view: View,
    ) -> AnnotationQualityAuditReport:
        """
        Convenience entry point to audit a single COCO JSON structure.
        """

        return self.audit([(data, dataset, view)])

    def audit_file(
        self,
        annotation_file: Path,
        dataset: DatasetType,
        view: View,
    ) -> AnnotationQualityAuditReport:
        """
        Read and audit one COCO JSON file, without modifying it.
        """

        import json

        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        return self.audit_json(data, dataset, view)

    # =====================================================
    # Per-file audit
    # =====================================================

    def _audit_single(
        self,
        data: dict[str, Any],
        dataset: DatasetType,
        view: View,
    ) -> list[ImageQualityAuditRecord]:

        if "images" not in data or not isinstance(data["images"], list):
            raise ValueError(
                "Invalid COCO structure: missing or malformed 'images'."
            )

        if "annotations" not in data or not isinstance(
            data["annotations"], list
        ):
            raise ValueError(
                "Invalid COCO structure: missing or malformed 'annotations'."
            )

        folder = ImageFolder(
            dataset=dataset,
            folder_name=view.value,
            view=view,
            path=Path("."),
        )

        records: dict[int, ImageQualityAuditRecord] = {}
        order: list[int] = []

        for item in data["images"]:

            image_id = item.get("id")

            if image_id in records:
                continue

            width = item.get("width")
            height = item.get("height")
            file_name = item.get("file_name")

            record = ImageQualityAuditRecord(
                image_id=image_id,
                file_name=file_name,
                width=width,
                height=height,
                dataset=dataset,
                view=view,
            )

            self._attach_identity(record, folder, file_name)

            records[image_id] = record
            order.append(image_id)

        for item in data["annotations"]:

            image_id = item.get("image_id")

            record = records.get(image_id)

            if record is None:
                # Orphan annotation: not tied to any known image in this
                # file. Out of scope for a per-image audit.
                continue

            record.annotations.append(
                self._audit_bbox(item, record.width, record.height)
            )

        for record in records.values():
            self._finalize_image_status(record)

        return [records[image_id] for image_id in order]

    # -----------------------------------------------------

    def _attach_identity(
        self,
        record: ImageQualityAuditRecord,
        folder: ImageFolder,
        file_name: str | None,
    ) -> None:

        if not file_name:

            record.parse_error = "Missing file_name."

            return

        try:

            parsed: ImageRecord = self._parser.parse(
                folder,
                Path(file_name),
            )

            record.animal_id = parsed.animal_id

            record.weight_kg = parsed.weight_kg

            record.sex = parsed.sex

        except Exception as ex:  # noqa: BLE001

            record.parse_error = str(ex)

    # -----------------------------------------------------

    def _audit_bbox(
        self,
        item: dict[str, Any],
        width: int | None,
        height: int | None,
    ) -> BBoxAuditRecord:

        annotation_id = item.get("id")

        bbox_raw = item.get("bbox")

        issues: list[str] = []

        if (
            not isinstance(bbox_raw, (list, tuple))
            or len(bbox_raw) != 4
        ):

            return BBoxAuditRecord(
                annotation_id=annotation_id,
                bbox_xywh=tuple(bbox_raw) if isinstance(
                    bbox_raw, (list, tuple)
                ) else bbox_raw,
                bbox_xyxy=None,
                status=AnnotationStatus.INVALID_BBOX,
                issues=["bbox is missing or does not have four values."],
            )

        x, y, w, h = bbox_raw

        bbox_xywh = (x, y, w, h)

        if all(self._is_zero(value) for value in bbox_xywh):

            return BBoxAuditRecord(
                annotation_id=annotation_id,
                bbox_xywh=bbox_xywh,
                bbox_xyxy=None,
                status=AnnotationStatus.ZERO_BBOX,
                issues=["bbox is exactly [0, 0, 0, 0]."],
            )

        if not all(self._is_finite(value) for value in bbox_xywh):

            issues.append("bbox contains non-finite (NaN/Inf) values.")

            return BBoxAuditRecord(
                annotation_id=annotation_id,
                bbox_xywh=bbox_xywh,
                bbox_xyxy=None,
                status=AnnotationStatus.INVALID_BBOX,
                issues=issues,
            )

        if w <= 0 or h <= 0:

            issues.append(
                f"non-positive dimensions: width={w}, height={h}."
            )

            return BBoxAuditRecord(
                annotation_id=annotation_id,
                bbox_xywh=bbox_xywh,
                bbox_xyxy=None,
                status=AnnotationStatus.INVALID_BBOX,
                issues=issues,
            )

        x1, y1, x2, y2 = x, y, x + w, y + h

        bbox_xyxy = (x1, y1, x2, y2)

        if width is not None and height is not None:

            if x1 < 0 or y1 < 0 or x2 > width or y2 > height:

                issues.append(
                    "bbox extends outside image boundaries "
                    f"({width}x{height})."
                )

                return BBoxAuditRecord(
                    annotation_id=annotation_id,
                    bbox_xywh=bbox_xywh,
                    bbox_xyxy=bbox_xyxy,
                    status=AnnotationStatus.OUT_OF_BOUNDS,
                    issues=issues,
                )

        return BBoxAuditRecord(
            annotation_id=annotation_id,
            bbox_xywh=bbox_xywh,
            bbox_xyxy=bbox_xyxy,
            status=AnnotationStatus.VALID,
            issues=issues,
        )

    @staticmethod
    def _is_zero(value: Any) -> bool:

        try:
            return float(value) == 0.0
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _is_finite(value: Any) -> bool:

        try:
            return math.isfinite(float(value))
        except (TypeError, ValueError):
            return False

    # -----------------------------------------------------

    def _finalize_image_status(
        self,
        record: ImageQualityAuditRecord,
    ) -> None:

        if record.annotation_count == 0:

            record.status = AnnotationStatus.MISSING

            record.issues = ["No annotation exists for this image."]

            return

        if record.annotation_count > 1:

            record.status = AnnotationStatus.MULTIPLE_ANNOTATIONS

            record.issues = [
                f"{record.annotation_count} annotations reference this "
                "image."
            ]

            return

        only = record.annotations[0]

        record.status = only.status

        record.issues = list(only.issues)

    # =====================================================
    # Animal-level grouping
    # =====================================================

    def _group_by_animal(
        self,
        images: list[ImageQualityAuditRecord],
    ) -> list[AnimalQualityAuditRecord]:

        grouped: dict[tuple[str, str], AnimalQualityAuditRecord] = {}

        order: list[tuple[str, str]] = []

        for image in images:

            if not image.animal_id:
                # Cannot reliably group unparsed animals; they remain in
                # the per-image report but are excluded from animal-level
                # pairing.
                continue

            dataset_key = image.dataset.value if image.dataset else ""

            key = (dataset_key, image.animal_id)

            if key not in grouped:

                grouped[key] = AnimalQualityAuditRecord(
                    animal_id=image.animal_id,
                    dataset=image.dataset,
                    weight_kg=image.weight_kg,
                    sex=image.sex,
                )

                order.append(key)

            animal = grouped[key]

            if image.view is View.SIDE:
                animal.side_images.append(image)
            elif image.view is View.REAR:
                animal.rear_images.append(image)

        animals = [grouped[key] for key in order]

        for animal in animals:
            animal.pairing_status = self._pairing_status(animal)

        return animals

    @staticmethod
    def _pairing_status(
        animal: AnimalQualityAuditRecord,
    ) -> AnimalPairingStatus:

        has_side = animal.has_valid_side
        has_rear = animal.has_valid_rear

        if has_side and has_rear:
            return AnimalPairingStatus.SIDE_REAR_VALID

        if has_side:
            return AnimalPairingStatus.SIDE_ONLY

        if has_rear:
            return AnimalPairingStatus.REAR_ONLY

        return AnimalPairingStatus.NO_VALID_VIEW

    # =====================================================
    # Summary
    # =====================================================

    def _summarize(
        self,
        images: list[ImageQualityAuditRecord],
        animals: list[AnimalQualityAuditRecord],
    ) -> AnnotationQualitySummary:

        summary = AnnotationQualitySummary()

        summary.total_images = len(images)

        summary.total_animals = len(animals)

        for image in images:

            summary.total_annotations += image.annotation_count

            if image.dataset is not None:

                key = image.dataset.value

                summary.by_dataset[key] = (
                    summary.by_dataset.get(key, 0) + 1
                )

            if image.view is not None:

                key = image.view.value

                summary.by_view[key] = (
                    summary.by_view.get(key, 0) + 1
                )

            status_key = image.status.value

            summary.by_status[status_key] = (
                summary.by_status.get(status_key, 0) + 1
            )

        for animal in animals:

            key = animal.pairing_status.value

            summary.by_pairing[key] = (
                summary.by_pairing.get(key, 0) + 1
            )

        return summary
