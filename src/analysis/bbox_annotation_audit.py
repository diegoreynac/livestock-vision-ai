"""
BBox annotation audit.

Read-only, per-image audit of the raw COCO annotation structure used to
answer one specific thesis question:

    "Can images without an available BBox but with valid keypoint
    annotations be used to derive a bounding box for supervised training?"

This module deliberately mirrors the read-only philosophy already used by
:mod:`src.coco.audit` (``COCOAudit``): it parses the COCO JSON directly and
does not depend on :class:`src.coco.reader.COCOReader` or a
``LivestockDataset``, because ``COCOReader`` already *rejects* annotations
whose bbox is missing/invalid (see ``COCOReader._create_annotation``) and
therefore cannot be used to inspect exactly the population we need to audit
here.

It does, however, reuse the project's existing COCO vocabulary instead of
re-inventing it:

    * :class:`src.coco.enums.KeypointVisibility` for the visibility flag.

No bounding box is reconstructed here. This module only determines, per
image, whether a bbox theoretically *could* be derived from valid
keypoints -- the actual derivation is left to a future, separate module.
"""

from __future__ import annotations

import json
import statistics
from collections import Counter
from collections.abc import Iterable
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.coco.enums import KeypointVisibility
from src.evaluation.detection import calculate_iou


# ==========================================================
# Configuration
# ==========================================================

# Minimum number of keypoints with finite, labeled coordinates required to
# consider a bounding box theoretically derivable. Two distinct labeled
# points are the minimum needed to span a non-degenerate box; this is a
# conservative structural check, not a claim about training suitability.
MIN_KEYPOINTS_FOR_BBOX = 2


# ==========================================================
# Classification
# ==========================================================

class ImageAnnotationClass(Enum):
    """
    Mutually exclusive classification of one COCO image.
    """

    # No COCO annotation object references this image at all.
    ANNOTATION_MISSING = "annotation_missing"

    # Annotation(s) exist and at least one has a structurally valid bbox.
    BBOX_AVAILABLE = "bbox_available"

    # Annotation(s) exist, no valid bbox, but keypoints look sufficient
    # to theoretically derive one.
    BBOX_MISSING_KEYPOINTS_AVAILABLE = "bbox_missing_keypoints_available"

    # Annotation(s) exist, no valid bbox, and keypoints are missing or
    # insufficient to derive one.
    BBOX_MISSING_KEYPOINTS_MISSING = "bbox_missing_keypoints_missing"

    # Annotation(s) exist and a bbox field is present but structurally
    # invalid (wrong length, non-numeric, non-positive size, etc.).
    BBOX_INVALID = "bbox_invalid"

    # Reserved for cases discovered in real data that do not fit the
    # buckets above (e.g. an unexpected annotation payload shape).
    OTHER = "other"


# ==========================================================
# Per-annotation keypoint audit
# ==========================================================

@dataclass(slots=True)
class KeypointAuditEntry:
    """
    Single keypoint reported for a specific annotation.
    """

    index: int
    x: float | None
    y: float | None
    visibility: int | None
    is_finite: bool
    is_labeled: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================================
# Candidate bounding box (derived, temporary, never written back)
# ==========================================================

@dataclass(slots=True, frozen=True)
class CandidateBoundingBoxXYXY:
    """
    Immutable, TEMPORARY candidate bounding box expressed in XYXY
    (``x_min, y_min, x_max, y_max``) form, derived purely from valid
    keypoint coordinates.

    This value never mutates the original annotation and is not written
    back into any COCO structure. It exists solely to support the
    evidence-gathering comparison against ground-truth BBoxes (via IoU).
    """

    x_min: float
    y_min: float
    x_max: float
    y_max: float
    source_keypoint_count: int

    @property
    def width(self) -> float:
        return self.x_max - self.x_min

    @property
    def height(self) -> float:
        return self.y_max - self.y_min

    @property
    def as_xywh(self) -> tuple[float, float, float, float]:
        """
        Return the candidate as COCO-style ``(x, y, width, height)`` so
        it can be compared with :func:`src.evaluation.detection.calculate_iou`.
        """

        return (self.x_min, self.y_min, self.width, self.height)

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["width"] = self.width
        result["height"] = self.height
        return result


def derive_candidate_bbox_from_keypoints(
    keypoints: list[KeypointAuditEntry],
) -> CandidateBoundingBoxXYXY | None:
    """
    Derive a TEMPORARY candidate XYXY bounding box from valid keypoints.

    A keypoint is usable only if it is:
        * finite (non-NaN, non-infinite numeric x/y), and
        * labeled, per the project's existing visibility convention
          (:class:`src.coco.enums.KeypointVisibility`: any value other
          than ``NOT_LABELED`` counts, i.e. both ``OCCLUDED`` and
          ``VISIBLE`` keypoints are used -- occluded-but-labeled points
          still carry a real, annotator-provided coordinate).

    At least :data:`MIN_KEYPOINTS_FOR_BBOX` usable keypoints are required
    to produce a "meaningful" box (a single point, or zero points, cannot
    span a rectangle). Returns ``None`` when derivation is not possible.

    This never mutates ``keypoints`` or any annotation; the return value
    is a new, independent, immutable object.
    """

    usable_points = [
        (entry.x, entry.y)
        for entry in keypoints
        if entry.is_finite and entry.is_labeled
    ]

    if len(usable_points) < MIN_KEYPOINTS_FOR_BBOX:
        return None

    xs = [x for x, _ in usable_points]
    ys = [y for _, y in usable_points]

    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)

    # A degenerate box (zero width or height, e.g. all usable points
    # collapse onto a single coordinate) is not meaningful for training.
    if x_max <= x_min or y_max <= y_min:
        return None

    return CandidateBoundingBoxXYXY(
        x_min=x_min,
        y_min=y_min,
        x_max=x_max,
        y_max=y_max,
        source_keypoint_count=len(usable_points),
    )


@dataclass(slots=True)
class AnnotationAudit:
    """
    Detailed, read-only audit of a single COCO annotation object.
    """

    annotation_id: int | None
    category_id: int | None
    category_name: str | None

    has_bbox: bool
    bbox_values: tuple[float, float, float, float] | None
    bbox_is_valid: bool

    has_keypoints: bool
    keypoint_count: int
    keypoints: list[KeypointAuditEntry] = field(default_factory=list)
    valid_keypoint_count: int = 0

    has_segmentation: bool = False
    segmentation_type: str | None = None

    # Populated separately by BBoxCandidateDeriver; kept optional here so
    # the base structural audit remains unaffected by this iteration.
    candidate_bbox: CandidateBoundingBoxXYXY | None = None
    iou_with_ground_truth: float | None = None

    @property
    def keypoints_sufficient_for_bbox(self) -> bool:
        """
        Whether valid keypoints are structurally sufficient to
        theoretically derive a bbox. No bbox is reconstructed here.
        """

        return self.valid_keypoint_count >= MIN_KEYPOINTS_FOR_BBOX

    def to_dict(self) -> dict[str, Any]:
        result = {
            "annotation_id": self.annotation_id,
            "category_id": self.category_id,
            "category_name": self.category_name,
            "has_bbox": self.has_bbox,
            "bbox_values": self.bbox_values,
            "bbox_is_valid": self.bbox_is_valid,
            "has_keypoints": self.has_keypoints,
            "keypoint_count": self.keypoint_count,
            "keypoints": [entry.to_dict() for entry in self.keypoints],
            "valid_keypoint_count": self.valid_keypoint_count,
            "has_segmentation": self.has_segmentation,
            "segmentation_type": self.segmentation_type,
            "keypoints_sufficient_for_bbox": self.keypoints_sufficient_for_bbox,
            "candidate_bbox": (
                self.candidate_bbox.to_dict()
                if self.candidate_bbox is not None
                else None
            ),
            "iou_with_ground_truth": self.iou_with_ground_truth,
        }
        return result


# ==========================================================
# Per-image audit
# ==========================================================

@dataclass(slots=True)
class ImageAnnotationAudit:
    """
    Detailed, read-only audit of a single COCO image entry and every
    annotation that references it.
    """

    source_file: str | None
    image_id: int | None
    filename: str | None
    width: int | None
    height: int | None

    annotations: list[AnnotationAudit] = field(default_factory=list)

    classification: ImageAnnotationClass = ImageAnnotationClass.ANNOTATION_MISSING

    @property
    def annotation_count(self) -> int:
        return len(self.annotations)

    @property
    def annotation_ids(self) -> list[int | None]:
        return [item.annotation_id for item in self.annotations]

    @property
    def category_ids(self) -> list[int | None]:
        return [item.category_id for item in self.annotations]

    @property
    def category_names(self) -> list[str | None]:
        return [item.category_name for item in self.annotations]

    @property
    def has_annotations(self) -> bool:
        return self.annotation_count > 0

    @property
    def has_valid_bbox(self) -> bool:
        return any(item.bbox_is_valid for item in self.annotations)

    @property
    def has_keypoints(self) -> bool:
        return any(item.has_keypoints for item in self.annotations)

    @property
    def keypoints_sufficient_for_bbox(self) -> bool:
        """
        True if at least one annotation on this image has keypoints that
        are structurally sufficient to theoretically derive a bbox.
        """

        return any(
            item.keypoints_sufficient_for_bbox for item in self.annotations
        )

    def get_annotation(self, annotation_id: int) -> AnnotationAudit | None:
        """
        Return one annotation audit entry by its ``annotation_id``.
        """

        for annotation in self.annotations:
            if annotation.annotation_id == annotation_id:
                return annotation

        return None

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "image_id": self.image_id,
            "filename": self.filename,
            "width": self.width,
            "height": self.height,
            "annotation_count": self.annotation_count,
            "annotation_ids": self.annotation_ids,
            "category_ids": self.category_ids,
            "category_names": self.category_names,
            "has_annotations": self.has_annotations,
            "has_valid_bbox": self.has_valid_bbox,
            "has_keypoints": self.has_keypoints,
            "keypoints_sufficient_for_bbox": self.keypoints_sufficient_for_bbox,
            "classification": self.classification.value,
            "annotations": [item.to_dict() for item in self.annotations],
        }


# ==========================================================
# Aggregate statistics
# ==========================================================

@dataclass(slots=True)
class BBoxAnnotationAuditStatistics:
    """
    Aggregate, image-identity-preserving statistics for the audit.
    """

    total_images: int = 0
    # A unique image identity is scoped to its source file because numeric
    # COCO image IDs may legitimately overlap across independently exported
    # JSON files.
    total_unique_images: int = 0
    unique_image_id_values: int = 0
    annotations_per_image: dict[int, int] = field(default_factory=dict)
    images_with_multiple_annotations: int = 0

    images_with_annotations: int = 0
    images_without_annotations: int = 0

    total_annotations: int = 0
    # ``total_annotations`` counts every record in the source JSON, while
    # ``indexed_annotations`` excludes records whose image_id is absent from
    # that JSON's images collection. Per-image reports cover indexed records.
    indexed_annotations: int = 0
    orphan_annotations: int = 0
    annotations_with_bbox: int = 0
    annotations_without_bbox: int = 0
    annotations_with_invalid_bbox: int = 0
    raw_annotations_with_valid_bbox: int = 0
    raw_annotations_with_invalid_bbox: int = 0

    annotations_with_keypoints: int = 0
    annotations_without_keypoints: int = 0

    annotations_missing_bbox_with_keypoints: int = 0
    annotations_keypoints_sufficient_for_bbox: int = 0

    invalid_or_incomplete_annotations: int = 0

    classification_counts: dict[str, int] = field(default_factory=dict)

    # --------------------------------------------------
    # Candidate BBox derivation (from keypoints)
    # --------------------------------------------------

    # Annotations that already have a valid ground-truth bbox AND valid
    # (labeled, finite) keypoints -- the population used to validate the
    # candidate-derivation method via IoU.
    annotations_with_valid_bbox_and_keypoints: int = 0

    # Annotations whose ground-truth bbox is invalid/absent but whose
    # keypoints are structurally usable to attempt a candidate derivation.
    annotations_with_invalid_bbox_and_usable_keypoints: int = 0

    # Number of annotations for which a candidate bbox was actually
    # produced (regardless of whether a ground-truth bbox exists).
    candidate_bboxes_derived: int = 0

    # IoU statistics, computed only for annotations where BOTH a valid
    # ground-truth bbox and a derived candidate bbox exist.
    iou_sample_count: int = 0
    iou_min: float | None = None
    iou_max: float | None = None
    iou_mean: float | None = None
    iou_median: float | None = None
    iou_stdev: float | None = None
    iou_q1: float | None = None
    iou_q3: float | None = None

    # Descriptive counts around commonly used IoU reference points
    # (0.5 / 0.75). These are NOT acceptance thresholds -- none currently
    # exists in the repository for this purpose -- they are reported only
    # to help gather evidence.
    iou_below_0_5: int = 0
    iou_at_or_above_0_5: int = 0
    iou_below_0_75: int = 0
    iou_at_or_above_0_75: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ==========================================================
# Report
# ==========================================================

@dataclass(slots=True)
class BBoxAnnotationAuditReport:
    """
    Full, image-identity-preserving audit report.
    """

    source_file: str | None = None
    images: list[ImageAnnotationAudit] = field(default_factory=list)
    statistics: BBoxAnnotationAuditStatistics = field(
        default_factory=BBoxAnnotationAuditStatistics
    )

    def get_image(self, image_id: int) -> ImageAnnotationAudit | None:
        """
        Return the detailed audit entry for a specific image_id, or None
        if the image_id was not present in the source JSON.
        """

        for image in self.images:
            if image.image_id == image_id:
                return image

        return None

    def images_by_classification(
        self,
        classification: ImageAnnotationClass,
    ) -> list[ImageAnnotationAudit]:
        """
        Return every image matching one classification bucket.
        """

        return [
            image
            for image in self.images
            if image.classification is classification
        ]

    def get_annotation(
        self,
        image_id: int,
        annotation_id: int,
    ) -> AnnotationAudit | None:
        """
        Return one annotation audit entry for direct inspection, given
        its image_id and annotation_id.
        """

        image = self.get_image(image_id)

        if image is None:
            return None

        return image.get_annotation(annotation_id)

    def inspect_annotation(
        self,
        image_id: int,
        annotation_id: int,
    ) -> dict[str, Any] | None:
        """
        Return a flattened inspection view of a single annotation,
        combining image-level identity with annotation-level detail:
        image_id, filename, image dimensions, annotation_id, original
        bbox, keypoints, candidate bbox, and IoU (when a ground-truth
        bbox exists).
        """

        image = self.get_image(image_id)

        if image is None:
            return None

        annotation = image.get_annotation(annotation_id)

        if annotation is None:
            return None

        return {
            "image_id": image.image_id,
            "filename": image.filename,
            "width": image.width,
            "height": image.height,
            "annotation_id": annotation.annotation_id,
            "bbox_values": annotation.bbox_values,
            "bbox_is_valid": annotation.bbox_is_valid,
            "keypoints": [entry.to_dict() for entry in annotation.keypoints],
            "candidate_bbox": (
                annotation.candidate_bbox.to_dict()
                if annotation.candidate_bbox is not None
                else None
            ),
            "iou_with_ground_truth": annotation.iou_with_ground_truth,
        }

    def annotations_with_candidate_from_invalid_bbox(
        self,
    ) -> list[tuple[ImageAnnotationAudit, AnnotationAudit]]:
        """
        Return (image, annotation) pairs where the ground-truth bbox is
        invalid/absent, keypoints are available, and a candidate bbox
        could potentially be derived. Intended for the "evidence" report
        requested for supervised-training feasibility analysis.
        """

        results: list[tuple[ImageAnnotationAudit, AnnotationAudit]] = []

        for image in self.images:
            for annotation in image.annotations:
                if annotation.bbox_is_valid:
                    continue

                if not annotation.has_keypoints:
                    continue

                if annotation.candidate_bbox is None:
                    continue

                results.append((image, annotation))

        return results

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_file": self.source_file,
            "statistics": self.statistics.to_dict(),
            "images": [image.to_dict() for image in self.images],
        }


@dataclass(slots=True, frozen=True)
class IoUAnnotationCase:
    """
    Immutable inspection record for one annotation in a descriptive IoU
    range. It retains source-scoped image identity for cross-file audits.
    """

    source_file: str | None
    image_id: int | None
    annotation_id: int | None
    filename: str | None
    width: int | None
    height: int | None
    ground_truth_bbox: tuple[float, float, float, float]
    candidate_bbox: CandidateBoundingBoxXYXY
    usable_keypoint_count: int
    iou: float

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["candidate_bbox"] = self.candidate_bbox.to_dict()
        return result


@dataclass(slots=True)
class BBoxAnnotationAuditCollectionReport:
    """
    Source-aware aggregate of independently audited COCO JSON files.

    ``image_id`` alone is not globally unique across COCO exports, so every
    aggregate image identity is the pair ``(source_file, image_id)``.
    """

    files: list[BBoxAnnotationAuditReport] = field(default_factory=list)
    statistics: BBoxAnnotationAuditStatistics = field(
        default_factory=BBoxAnnotationAuditStatistics
    )

    def inspect_annotation(
        self,
        source_file: str,
        image_id: int,
        annotation_id: int,
    ) -> dict[str, Any] | None:
        """Return a source-scoped annotation inspection record."""

        for report in self.files:
            if report.source_file == source_file:
                return report.inspect_annotation(image_id, annotation_id)

        return None

    def iou_cases_below(
        self,
        upper_bound: float,
    ) -> list[IoUAnnotationCase]:
        """Return cases whose measured IoU is strictly below ``upper_bound``."""

        return [
            case
            for case in self._iou_cases()
            if case.iou < upper_bound
        ]

    def iou_cases_in_range(
        self,
        lower_bound: float,
        upper_bound: float,
    ) -> list[IoUAnnotationCase]:
        """
        Return cases satisfying ``lower_bound <= IoU < upper_bound``.

        The caller supplies descriptive bounds; this method does not define
        an acceptance threshold.
        """

        return [
            case
            for case in self._iou_cases()
            if lower_bound <= case.iou < upper_bound
        ]

    def _iou_cases(self) -> list[IoUAnnotationCase]:
        cases: list[IoUAnnotationCase] = []

        for report in self.files:
            for image in report.images:
                for annotation in image.annotations:
                    if annotation.iou_with_ground_truth is None:
                        continue

                    assert annotation.bbox_values is not None
                    assert annotation.candidate_bbox is not None

                    cases.append(
                        IoUAnnotationCase(
                            source_file=report.source_file,
                            image_id=image.image_id,
                            annotation_id=annotation.annotation_id,
                            filename=image.filename,
                            width=image.width,
                            height=image.height,
                            ground_truth_bbox=annotation.bbox_values,
                            candidate_bbox=annotation.candidate_bbox,
                            usable_keypoint_count=annotation.valid_keypoint_count,
                            iou=annotation.iou_with_ground_truth,
                        )
                    )

        return cases

    def to_dict(self) -> dict[str, Any]:
        return {
            "statistics": self.statistics.to_dict(),
            "files": [report.to_dict() for report in self.files],
        }


# ==========================================================
# Audit
# ==========================================================

class BBoxAnnotationAudit:
    """
    Read-only, per-image audit of a COCO annotation JSON structure.

    Answers, without modifying anything:

        A) images with no annotations at all
        B) images with annotations but no available bbox
        C) images with annotations and a valid bbox
        D) images with annotations, no bbox, but keypoints that are
           structurally sufficient to theoretically derive one
    """

    _REQUIRED_TOP_LEVEL_KEYS = ("images", "annotations")

    def audit(
        self,
        data: dict[str, Any],
        source_file: str | None = None,
    ) -> BBoxAnnotationAuditReport:
        """
        Audit an already-parsed COCO JSON dictionary.

        ``source_file`` is optional for unit-sized in-memory data, but should
        be supplied for multi-file audits to preserve source-scoped identity.
        """

        for key in self._REQUIRED_TOP_LEVEL_KEYS:
            if key not in data or not isinstance(data[key], list):
                raise ValueError(
                    f"Invalid COCO structure: missing or malformed '{key}'."
                )

        categories = self._read_categories(data.get("categories", []))

        images_by_id: dict[int, ImageAnnotationAudit] = {}
        order: list[int] = []

        for item in data["images"]:
            image_id = item.get("id")

            if image_id in images_by_id:
                continue

            images_by_id[image_id] = ImageAnnotationAudit(
                source_file=source_file,
                image_id=image_id,
                filename=item.get("file_name"),
                width=item.get("width"),
                height=item.get("height"),
            )
            order.append(image_id)

        for item in data["annotations"]:
            image_id = item.get("image_id")
            image_audit = images_by_id.get(image_id)

            if image_audit is None:
                # Orphan annotation: not tied to any known image. Out of
                # scope for a per-image audit, so it is skipped here; the
                # existing COCOAudit already reports orphan counts.
                continue

            image_audit.annotations.append(
                self._audit_annotation(item, categories)
            )

        report_images = [images_by_id[image_id] for image_id in order]

        for image_audit in report_images:
            image_audit.classification = self._classify(image_audit)

        for image_audit in report_images:
            for annotation in image_audit.annotations:
                self._derive_candidate_and_iou(annotation)

        statistics = self._aggregate(report_images)
        statistics.total_annotations = len(data["annotations"])
        statistics.indexed_annotations = sum(
            image.annotation_count for image in report_images
        )
        statistics.orphan_annotations = (
            statistics.total_annotations - statistics.indexed_annotations
        )
        for item in data["annotations"]:
            _, bbox_is_valid = self._parse_bbox(item.get("bbox"))
            if bbox_is_valid:
                statistics.raw_annotations_with_valid_bbox += 1
            else:
                statistics.raw_annotations_with_invalid_bbox += 1

        return BBoxAnnotationAuditReport(
            source_file=source_file,
            images=report_images,
            statistics=statistics,
        )

    def audit_file(
        self,
        annotation_file: Path,
    ) -> BBoxAnnotationAuditReport:
        """
        Read and audit one COCO JSON file without modifying it.
        """

        with annotation_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError(
                f"Invalid COCO structure in {annotation_file}: top-level "
                "JSON value must be an object."
            )

        return self.audit(data, source_file=str(annotation_file))

    def audit_files(
        self,
        annotation_files: Iterable[Path],
    ) -> BBoxAnnotationAuditCollectionReport:
        """
        Audit multiple COCO JSON files and calculate source-aware aggregate
        statistics. Numeric image IDs are never treated as globally unique.
        """

        reports = [
            self.audit_file(annotation_file)
            for annotation_file in annotation_files
        ]

        return BBoxAnnotationAuditCollectionReport(
            files=reports,
            statistics=self._aggregate_collection(reports),
        )

    # --------------------------------------------------
    # Per-annotation audit
    # --------------------------------------------------

    def _audit_annotation(
        self,
        item: dict[str, Any],
        categories: dict[int, str],
    ) -> AnnotationAudit:

        category_id = item.get("category_id")

        bbox_raw = item.get("bbox")
        has_bbox = bbox_raw is not None
        bbox_values, bbox_is_valid = self._parse_bbox(bbox_raw)

        keypoints_raw = item.get("keypoints", [])
        keypoint_entries = self._parse_keypoints(keypoints_raw)
        valid_keypoint_count = sum(
            1
            for entry in keypoint_entries
            if entry.is_finite and entry.is_labeled
        )

        segmentation_raw = item.get("segmentation")
        has_segmentation, segmentation_type = self._describe_segmentation(
            segmentation_raw
        )

        return AnnotationAudit(
            annotation_id=item.get("id"),
            category_id=category_id,
            category_name=categories.get(category_id),
            has_bbox=has_bbox,
            bbox_values=bbox_values,
            bbox_is_valid=bbox_is_valid,
            has_keypoints=len(keypoint_entries) > 0,
            keypoint_count=len(keypoint_entries),
            keypoints=keypoint_entries,
            valid_keypoint_count=valid_keypoint_count,
            has_segmentation=has_segmentation,
            segmentation_type=segmentation_type,
        )

    @staticmethod
    def _read_categories(items: list[dict[str, Any]]) -> dict[int, str]:
        categories: dict[int, str] = {}

        for item in items:
            category_id = item.get("id")
            name = item.get("name")

            if category_id is not None:
                categories[category_id] = name

        return categories

    @staticmethod
    def _parse_bbox(
        bbox_raw: Any,
    ) -> tuple[tuple[float, float, float, float] | None, bool]:
        """
        Parse a raw ``bbox`` field.

        Returns the parsed 4-tuple (when the shape allows it) and whether
        it is a structurally valid COCO bbox (finite numeric values with
        a strictly positive width/height).
        """

        if bbox_raw is None:
            return None, False

        if not isinstance(bbox_raw, list) or len(bbox_raw) != 4:
            return None, False

        if not all(isinstance(value, (int, float)) for value in bbox_raw):
            return None, False

        x, y, width, height = bbox_raw

        if not all(_is_finite(value) for value in (x, y, width, height)):
            return (x, y, width, height), False

        is_valid = x >= 0 and y >= 0 and width > 0 and height > 0

        return (x, y, width, height), is_valid

    @staticmethod
    def _parse_keypoints(values: Any) -> list[KeypointAuditEntry]:
        """
        Parse a raw ``keypoints`` field following COCO's flat
        ``[x, y, visibility] * N`` representation.

        Malformed entries (wrong cardinality, non-numeric coordinates,
        or an unrecognized visibility flag) are still reported, marked as
        not finite/not labeled, instead of being silently discarded --
        this audit must be able to show *why* a bbox cannot be derived.
        """

        entries: list[KeypointAuditEntry] = []

        if not isinstance(values, list) or len(values) % 3 != 0:
            return entries

        for index, start in enumerate(range(0, len(values), 3)):
            x, y, visibility_raw = values[start:start + 3]

            x_value = x if isinstance(x, (int, float)) else None
            y_value = y if isinstance(y, (int, float)) else None

            is_finite = (
                x_value is not None
                and y_value is not None
                and _is_finite(x_value)
                and _is_finite(y_value)
            )

            visibility: KeypointVisibility | None
            try:
                visibility = KeypointVisibility(visibility_raw)
            except (TypeError, ValueError):
                visibility = None

            entries.append(
                KeypointAuditEntry(
                    index=index,
                    x=x_value,
                    y=y_value,
                    visibility=int(visibility) if visibility is not None else None,
                    is_finite=is_finite,
                    is_labeled=bool(visibility is not None and visibility.is_labeled),
                )
            )

        return entries

    @staticmethod
    def _describe_segmentation(
        segmentation_raw: Any,
    ) -> tuple[bool, str | None]:

        if segmentation_raw is None:
            return False, None

        if isinstance(segmentation_raw, list):
            if len(segmentation_raw) == 0:
                return False, None
            return True, "polygon"

        if isinstance(segmentation_raw, dict):
            return True, "rle"

        return True, "unknown"

    # --------------------------------------------------
    # Candidate bbox derivation (from keypoints) + IoU
    # --------------------------------------------------

    @staticmethod
    def _derive_candidate_and_iou(annotation: AnnotationAudit) -> None:
        """
        Derive a TEMPORARY candidate XYXY bbox from this annotation's
        valid keypoints (if sufficient) and, when a valid ground-truth
        bbox also exists, compute the IoU between them.

        Mutates only the ``AnnotationAudit`` value object being audited
        (never the source COCO annotation dict); the derived candidate
        is a separate, immutable representation.
        """

        candidate = derive_candidate_bbox_from_keypoints(annotation.keypoints)
        annotation.candidate_bbox = candidate

        if candidate is None or not annotation.bbox_is_valid:
            return

        assert annotation.bbox_values is not None

        annotation.iou_with_ground_truth = calculate_iou(
            candidate.as_xywh,
            annotation.bbox_values,
        )

    # --------------------------------------------------
    # Classification
    # --------------------------------------------------

    @staticmethod
    def _classify(image: ImageAnnotationAudit) -> ImageAnnotationClass:

        if not image.has_annotations:
            return ImageAnnotationClass.ANNOTATION_MISSING

        if image.has_valid_bbox:
            return ImageAnnotationClass.BBOX_AVAILABLE

        # No valid bbox on any annotation. Distinguish an invalid (but
        # present) bbox payload from a genuinely absent one.
        has_invalid_bbox_payload = any(
            item.has_bbox and not item.bbox_is_valid
            for item in image.annotations
        )

        if has_invalid_bbox_payload:
            return ImageAnnotationClass.BBOX_INVALID

        if image.keypoints_sufficient_for_bbox:
            return ImageAnnotationClass.BBOX_MISSING_KEYPOINTS_AVAILABLE

        return ImageAnnotationClass.BBOX_MISSING_KEYPOINTS_MISSING

    # --------------------------------------------------
    # Aggregation
    # --------------------------------------------------

    @staticmethod
    def _aggregate(
        images: list[ImageAnnotationAudit],
    ) -> BBoxAnnotationAuditStatistics:

        stats = BBoxAnnotationAuditStatistics()
        stats.total_images = len(images)
        stats.total_unique_images = len(images)
        stats.unique_image_id_values = len(
            {
                image.image_id
                for image in images
            }
        )

        classification_counter: Counter[str] = Counter()
        iou_samples: list[float] = []

        for image in images:
            classification_counter[image.classification.value] += 1
            stats.annotations_per_image[image.annotation_count] = (
                stats.annotations_per_image.get(image.annotation_count, 0) + 1
            )

            if image.annotation_count > 1:
                stats.images_with_multiple_annotations += 1

            if image.has_annotations:
                stats.images_with_annotations += 1
            else:
                stats.images_without_annotations += 1

            for annotation in image.annotations:
                stats.indexed_annotations += 1

                if annotation.bbox_is_valid:
                    stats.annotations_with_bbox += 1
                else:
                    stats.annotations_without_bbox += 1

                if annotation.has_bbox and not annotation.bbox_is_valid:
                    stats.annotations_with_invalid_bbox += 1

                if annotation.has_keypoints:
                    stats.annotations_with_keypoints += 1
                else:
                    stats.annotations_without_keypoints += 1

                if not annotation.bbox_is_valid and annotation.has_keypoints:
                    stats.annotations_missing_bbox_with_keypoints += 1

                if (
                    not annotation.bbox_is_valid
                    and annotation.keypoints_sufficient_for_bbox
                ):
                    stats.annotations_keypoints_sufficient_for_bbox += 1

                if not annotation.bbox_is_valid and not annotation.has_keypoints:
                    stats.invalid_or_incomplete_annotations += 1

                if annotation.bbox_is_valid and annotation.keypoints_sufficient_for_bbox:
                    stats.annotations_with_valid_bbox_and_keypoints += 1

                if (
                    not annotation.bbox_is_valid
                    and annotation.keypoints_sufficient_for_bbox
                ):
                    stats.annotations_with_invalid_bbox_and_usable_keypoints += 1

                if annotation.candidate_bbox is not None:
                    stats.candidate_bboxes_derived += 1

                if annotation.iou_with_ground_truth is not None:
                    iou_samples.append(annotation.iou_with_ground_truth)

        stats.classification_counts = dict(classification_counter)
        _populate_iou_statistics(stats, iou_samples)

        return stats

    @staticmethod
    def _aggregate_collection(
        reports: list[BBoxAnnotationAuditReport],
    ) -> BBoxAnnotationAuditStatistics:
        """
        Aggregate file reports without collapsing same-numbered image IDs
        originating from different COCO JSON files.
        """

        images = [
            image
            for report in reports
            for image in report.images
        ]

        stats = BBoxAnnotationAudit._aggregate(images)
        stats.total_annotations = sum(
            report.statistics.total_annotations for report in reports
        )
        stats.indexed_annotations = sum(
            report.statistics.indexed_annotations for report in reports
        )
        stats.orphan_annotations = sum(
            report.statistics.orphan_annotations for report in reports
        )
        stats.raw_annotations_with_valid_bbox = sum(
            report.statistics.raw_annotations_with_valid_bbox
            for report in reports
        )
        stats.raw_annotations_with_invalid_bbox = sum(
            report.statistics.raw_annotations_with_invalid_bbox
            for report in reports
        )

        return stats


def _populate_iou_statistics(
    stats: BBoxAnnotationAuditStatistics,
    iou_samples: list[float],
) -> None:
    """
    Populate descriptive IoU statistics from the collected samples.

    Only annotations that had BOTH a valid ground-truth bbox and a
    successfully derived candidate bbox contribute a sample; this is
    the population needed to evaluate candidate-derivation reliability.
    """

    stats.iou_sample_count = len(iou_samples)

    if not iou_samples:
        return

    ordered = sorted(iou_samples)

    stats.iou_min = ordered[0]
    stats.iou_max = ordered[-1]
    stats.iou_mean = statistics.fmean(ordered)
    stats.iou_median = statistics.median(ordered)
    stats.iou_stdev = (
        statistics.stdev(ordered) if len(ordered) > 1 else 0.0
    )

    if len(ordered) >= 2:
        quantiles = statistics.quantiles(ordered, n=4, method="inclusive")
        stats.iou_q1 = quantiles[0]
        stats.iou_q3 = quantiles[2]
    else:
        stats.iou_q1 = ordered[0]
        stats.iou_q3 = ordered[0]

    for value in ordered:
        if value < 0.5:
            stats.iou_below_0_5 += 1
        else:
            stats.iou_at_or_above_0_5 += 1

        if value < 0.75:
            stats.iou_below_0_75 += 1
        else:
            stats.iou_at_or_above_0_75 += 1


def _is_finite(value: float) -> bool:
    """
    Return True for a numeric, finite (non-NaN, non-infinite) value.
    """

    try:
        return value == value and value not in (float("inf"), float("-inf"))
    except TypeError:
        return False
