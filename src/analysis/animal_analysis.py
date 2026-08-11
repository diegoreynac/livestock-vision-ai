"""Animal-level dataset analysis.

This module computes reproducible animal-level summaries from already-loaded
ImageRecord objects. It does not read image files or JSON files, and it does
not modify the existing dataset objects.
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Iterable, Protocol

from src.coco.models import COCOAnnotation
from src.dataset.models import ImageRecord


@dataclass(slots=True)
class KeypointSummary:
    """Internal summary for keypoint coverage."""

    available: bool
    count: int


class HasRecordsProtocol(Protocol):
    records: list[ImageRecord]


@dataclass(slots=True)
class AnimalRecordAnalysis:
    """Represents the analysis of a single animal."""

    animal_id: str
    image_count: int
    dataset_types: list[str]
    view_types: list[str]
    view_category: str

    weight_values: list[float]
    distinct_weight_values: list[float]
    consistent_weight: bool

    sex_values: list[str]
    distinct_sex_values: list[str]
    consistent_sex: bool

    annotation_image_count: int
    valid_bbox_image_count: int
    keypoint_image_count: int
    total_keypoints: int
    keypoint_cardinalities: list[int]

    has_annotation: bool
    has_valid_bbox: bool
    has_keypoints: bool

    def to_dict(self) -> dict[str, object]:
        """Serialize the animal analysis into a JSON-safe dictionary."""

        return {
            "animal_id": self.animal_id,
            "image_count": self.image_count,
            "dataset_types": list(self.dataset_types),
            "view_types": list(self.view_types),
            "view_category": self.view_category,
            "weight_values": list(self.weight_values),
            "distinct_weight_values": list(self.distinct_weight_values),
            "consistent_weight": self.consistent_weight,
            "sex_values": list(self.sex_values),
            "distinct_sex_values": list(self.distinct_sex_values),
            "consistent_sex": self.consistent_sex,
            "annotation_image_count": self.annotation_image_count,
            "valid_bbox_image_count": self.valid_bbox_image_count,
            "keypoint_image_count": self.keypoint_image_count,
            "total_keypoints": self.total_keypoints,
            "keypoint_cardinalities": list(self.keypoint_cardinalities),
            "has_annotation": self.has_annotation,
            "has_valid_bbox": self.has_valid_bbox,
            "has_keypoints": self.has_keypoints,
        }


@dataclass(slots=True)
class AnimalAnalysisReport:
    """Stores aggregated animal-level analysis results."""

    total_animals: int
    sex_distribution: dict[str, int]
    dataset_distribution: dict[str, int]
    view_distribution: dict[str, int]
    weight_distribution: dict[str, int]
    image_count_distribution: dict[int, int]
    annotation_coverage: dict[str, int]
    animals_with_multiple_views: int
    animals_with_inconsistent_weight: int
    animals_with_inconsistent_sex: int
    inconsistent_weight_animals: list[str]
    inconsistent_sex_animals: list[str]
    animals: list[AnimalRecordAnalysis]

    def to_dict(self) -> dict[str, object]:
        """Serialize the animal analysis report into a JSON-safe dictionary."""

        return {
            "total_animals": self.total_animals,
            "sex_distribution": dict(self.sex_distribution),
            "dataset_distribution": dict(self.dataset_distribution),
            "view_distribution": dict(self.view_distribution),
            "weight_distribution": dict(self.weight_distribution),
            "image_count_distribution": dict(self.image_count_distribution),
            "annotation_coverage": dict(self.annotation_coverage),
            "animals_with_multiple_views": self.animals_with_multiple_views,
            "animals_with_inconsistent_weight": self.animals_with_inconsistent_weight,
            "animals_with_inconsistent_sex": self.animals_with_inconsistent_sex,
            "inconsistent_weight_animals": list(self.inconsistent_weight_animals),
            "inconsistent_sex_animals": list(self.inconsistent_sex_animals),
            "animals": [animal.to_dict() for animal in self.animals],
        }


@dataclass(slots=True)
class AnimalAnalysis:
    """Analyzes livestock records at the animal level."""

    weight_bins: list[float] = field(default_factory=lambda: [0.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0])
    weight_bin_labels: list[str] = field(init=False)

    def __post_init__(self) -> None:
        self._validate_weight_bins(self.weight_bins)
        self.weight_bin_labels = self._build_weight_bin_labels(self.weight_bins)

    def analyze(self, records: Iterable[ImageRecord] | HasRecordsProtocol) -> AnimalAnalysisReport:
        """Compute the animal-level analysis report."""

        record_list = self._resolve_records(records)
        groups = self._group_by_animal(record_list)
        animals = [self._analyze_animal(animal_id, group) for animal_id, group in groups.items()]
        animals = sorted(animals, key=lambda item: item.animal_id)

        return AnimalAnalysisReport(
            total_animals=len(animals),
            sex_distribution=self._sex_distribution(animals),
            dataset_distribution=self._dataset_distribution(animals),
            view_distribution=self._view_distribution(animals),
            weight_distribution=self._weight_distribution(animals),
            image_count_distribution=self._image_count_distribution(animals),
            annotation_coverage=self._annotation_coverage(animals),
            animals_with_multiple_views=self._count_multiple_views(animals),
            animals_with_inconsistent_weight=self._count_inconsistent_weight(animals),
            animals_with_inconsistent_sex=self._count_inconsistent_sex(animals),
            inconsistent_weight_animals=[animal.animal_id for animal in animals if not animal.consistent_weight],
            inconsistent_sex_animals=[animal.animal_id for animal in animals if not animal.consistent_sex],
            animals=animals,
        )

    def _resolve_records(self, records: Iterable[ImageRecord] | HasRecordsProtocol) -> list[ImageRecord]:
        if hasattr(records, "records"):
            return list(records.records)

        return list(records)

    def _group_by_animal(self, records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
        groups: dict[str, list[ImageRecord]] = defaultdict(list)

        for record in records:
            groups[record.animal_id].append(record)

        return {animal_id: sorted(group, key=lambda record: record.filename) for animal_id, group in groups.items()}

    def _analyze_animal(self, animal_id: str, records: list[ImageRecord]) -> AnimalRecordAnalysis:
        view_types = self._unique_sorted_views(records)
        dataset_types = self._unique_sorted_datasets(records)
        weight_values = [record.weight_kg for record in records]
        distinct_weight_values = self._unique_sorted_weights(weight_values)
        consistent_weight = len(weight_values) > 0 and len(distinct_weight_values) == 1
        sex_values = [record.sex.value for record in records]
        distinct_sex_values = self._unique_sorted_sexes(sex_values)
        consistent_sex = len(sex_values) > 0 and len(distinct_sex_values) == 1
        annotation_image_count = sum(record.annotation is not None for record in records)
        valid_bbox_image_count = sum(self._has_valid_bbox(record.annotation) for record in records)
        keypoint_image_count = sum(self._has_keypoints(record.annotation) for record in records)
        total_keypoints = sum(self._keypoint_summary(record.annotation).count for record in records)
        keypoint_cardinalities = self._unique_sorted_keypoint_cardinalities(records)

        return AnimalRecordAnalysis(
            animal_id=animal_id,
            image_count=len(records),
            dataset_types=dataset_types,
            view_types=view_types,
            view_category=self._view_category(view_types),
            weight_values=weight_values,
            distinct_weight_values=distinct_weight_values,
            consistent_weight=consistent_weight,
            sex_values=sex_values,
            distinct_sex_values=distinct_sex_values,
            consistent_sex=consistent_sex,
            annotation_image_count=annotation_image_count,
            valid_bbox_image_count=valid_bbox_image_count,
            keypoint_image_count=keypoint_image_count,
            total_keypoints=total_keypoints,
            keypoint_cardinalities=keypoint_cardinalities,
            has_annotation=annotation_image_count > 0,
            has_valid_bbox=valid_bbox_image_count > 0,
            has_keypoints=keypoint_image_count > 0,
        )

    def _unique_sorted_views(self, records: list[ImageRecord]) -> list[str]:
        return sorted({record.view.value for record in records})

    def _unique_sorted_datasets(self, records: list[ImageRecord]) -> list[str]:
        return sorted({record.dataset.value for record in records})

    def _unique_sorted_weights(self, weights: list[float]) -> list[float]:
        return sorted(set(weights))

    def _unique_sorted_sexes(self, sexes: list[str]) -> list[str]:
        return sorted(set(sexes))

    def _unique_sorted_keypoint_cardinalities(self, records: list[ImageRecord]) -> list[int]:
        cardinalities = {
            self._keypoint_count(record.annotation)
            for record in records
            if record.annotation is not None
        }

        return sorted(cardinalities)

    def _view_category(self, view_types: list[str]) -> str:
        view_set = set(view_types)

        if view_set == {"Side"}:
            return "Side only"

        if view_set == {"Rear"}:
            return "Rear only"

        if view_set == {"Side", "Rear"}:
            return "Side + Rear"

        return "Unknown"

    def _has_valid_bbox(self, annotation: COCOAnnotation | None) -> bool:
        if annotation is None:
            return False

        bbox = annotation.bbox

        return (
            self._is_finite_number(bbox.x)
            and self._is_finite_number(bbox.y)
            and self._is_finite_number(bbox.width)
            and self._is_finite_number(bbox.height)
            and bbox.x >= 0.0
            and bbox.y >= 0.0
            and bbox.width > 0.0
            and bbox.height > 0.0
        )

    def _has_keypoints(self, annotation: COCOAnnotation | None) -> bool:
        return annotation is not None and annotation.has_keypoints

    def _keypoint_summary(self, annotation: COCOAnnotation | None) -> KeypointSummary:
        if annotation is None:
            return KeypointSummary(available=False, count=0)

        return KeypointSummary(
            available=annotation.has_keypoints,
            count=len(annotation.keypoints),
        )

    def _keypoint_count(self, annotation: COCOAnnotation | None) -> int:
        return self._keypoint_summary(annotation).count

    def _is_finite_number(self, value: float) -> bool:
        return isinstance(value, (float, int)) and math.isfinite(value)

    def _sex_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            if not animal.consistent_sex:
                counter["inconsistent"] += 1
            else:
                counter[animal.distinct_sex_values[0]] += 1

        return dict(counter)

    def _dataset_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            for dataset_type in animal.dataset_types:
                counter[dataset_type] += 1

        return dict(counter)

    def _view_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter(animal.view_category for animal in animals)
        return dict(counter)

    def _weight_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            if not animal.consistent_weight:
                counter["inconsistent"] += 1
                continue

            weight_value = animal.distinct_weight_values[0]
            counter[self._bin_weight(weight_value)] += 1

        return dict(counter)

    def _image_count_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[int, int]:
        counter = Counter(animal.image_count for animal in animals)
        return dict(sorted(counter.items()))

    def _annotation_coverage(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        animals_with_annotation = sum(animal.has_annotation for animal in animals)
        animals_with_valid_bbox = sum(animal.has_valid_bbox for animal in animals)
        animals_with_keypoints = sum(animal.has_keypoints for animal in animals)

        return {
            "animals_with_annotation": animals_with_annotation,
            "animals_without_annotation": len(animals) - animals_with_annotation,
            "animals_with_valid_bbox": animals_with_valid_bbox,
            "animals_with_keypoints": animals_with_keypoints,
        }

    def _count_multiple_views(self, animals: list[AnimalRecordAnalysis]) -> int:
        return sum(animal.view_category == "Side + Rear" for animal in animals)

    def _count_inconsistent_weight(self, animals: list[AnimalRecordAnalysis]) -> int:
        return sum(not animal.consistent_weight for animal in animals)

    def _count_inconsistent_sex(self, animals: list[AnimalRecordAnalysis]) -> int:
        return sum(not animal.consistent_sex for animal in animals)

    def _validate_weight_bins(self, weight_bins: list[float]) -> None:
        if len(weight_bins) < 2:
            raise ValueError("weight_bins must contain at least two values")

        if any(not isinstance(value, (int, float)) for value in weight_bins):
            raise TypeError("weight_bins values must be numeric")

        if any(weight_bins[i] >= weight_bins[i + 1] for i in range(len(weight_bins) - 1)):
            raise ValueError("weight_bins must be strictly increasing")

    def _build_weight_bin_labels(self, weight_bins: list[float]) -> list[str]:
        labels: list[str] = []

        for start, end in zip(weight_bins, weight_bins[1:]):
            labels.append(f"{int(start)}-{int(end)}")

        labels.append(f"{int(weight_bins[-1])}+")

        return labels

    def _bin_weight(self, weight: float) -> str:
        if weight < self.weight_bins[0]:
            return "below_minimum"

        for lower, upper in zip(self.weight_bins, self.weight_bins[1:]):
            if lower <= weight < upper:
                return f"{int(lower)}-{int(upper)}"

        return f"{int(self.weight_bins[-1])}+"
