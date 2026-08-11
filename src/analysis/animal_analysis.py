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

    weight_values: list[float | None]
    distinct_weight_values: list[float | None]
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
    total_images: int
    valid_image_count: int
    average_images_per_animal: float
    median_images_per_animal: float

    sex_distribution: dict[str, int]
    dataset_distribution: dict[str, int]
    dataset_membership_distribution: dict[str, int]
    dataset_exclusive_distribution: dict[str, int]
    view_distribution: dict[str, int]
    weight_distribution: dict[str, int]
    image_count_distribution: dict[int, int]
    annotation_coverage: dict[str, int]

    invalid_animal_id_records: int
    records_with_missing_folder: int
    records_with_missing_weight: int
    records_with_missing_sex: int

    animals_with_multiple_views: int
    animals_with_inconsistent_weight: int
    animals_with_inconsistent_sex: int
    inconsistent_weight_animals: list[str]
    inconsistent_sex_animals: list[str]
    animals: list[AnimalRecordAnalysis]
    metadata: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        """Serialize the animal analysis report into a JSON-safe dictionary."""

        return {
            "total_animals": self.total_animals,
            "total_images": self.total_images,
            "valid_image_count": self.valid_image_count,
            "average_images_per_animal": self.average_images_per_animal,
            "median_images_per_animal": self.median_images_per_animal,
            "sex_distribution": dict(self.sex_distribution),
            "dataset_distribution": dict(self.dataset_distribution),
            "dataset_membership_distribution": dict(self.dataset_membership_distribution),
            "dataset_exclusive_distribution": dict(self.dataset_exclusive_distribution),
            "view_distribution": dict(self.view_distribution),
            "weight_distribution": dict(self.weight_distribution),
            "image_count_distribution": dict(self.image_count_distribution),
            "annotation_coverage": dict(self.annotation_coverage),
            "invalid_animal_id_records": self.invalid_animal_id_records,
            "records_with_missing_folder": self.records_with_missing_folder,
            "records_with_missing_weight": self.records_with_missing_weight,
            "records_with_missing_sex": self.records_with_missing_sex,
            "animals_with_multiple_views": self.animals_with_multiple_views,
            "animals_with_inconsistent_weight": self.animals_with_inconsistent_weight,
            "animals_with_inconsistent_sex": self.animals_with_inconsistent_sex,
            "inconsistent_weight_animals": list(self.inconsistent_weight_animals),
            "inconsistent_sex_animals": list(self.inconsistent_sex_animals),
            "animals": [animal.to_dict() for animal in self.animals],
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class AnimalAnalysis:
    """Analyzes livestock records at the animal level."""

    weight_tolerance: float = 0.01
    weight_bins: list[float] = field(default_factory=lambda: [0.0, 100.0, 150.0, 200.0, 250.0, 300.0, 400.0])

    def __post_init__(self) -> None:
        self._validate_weight_tolerance(self.weight_tolerance)
        self._validate_weight_bins(self.weight_bins)

    def analyze(self, records: Iterable[ImageRecord] | HasRecordsProtocol) -> AnimalAnalysisReport:
        """Compute the animal-level analysis report."""

        record_list = self._resolve_records(records)
        invalid_animal_id_records = self._count_invalid_animal_id_records(record_list)
        groups, invalid_count = self._group_by_animal(record_list)

        animals = [self._analyze_animal(animal_id, group) for animal_id, group in groups.items()]
        animals = sorted(animals, key=lambda item: item.animal_id)

        total_images = len(record_list)
        valid_image_count = total_images - invalid_count
        average_images_per_animal = float(valid_image_count) / len(animals) if animals else 0.0
        median_images_per_animal = self._median_image_count(animals)

        return AnimalAnalysisReport(
            total_animals=len(animals),
            total_images=total_images,
            valid_image_count=valid_image_count,
            average_images_per_animal=average_images_per_animal,
            median_images_per_animal=median_images_per_animal,
            sex_distribution=self._sex_distribution(animals),
            dataset_distribution=self._dataset_distribution(animals),
            dataset_membership_distribution=self._dataset_distribution(animals),
            dataset_exclusive_distribution=self._dataset_exclusive_distribution(animals),
            view_distribution=self._view_distribution(animals),
            weight_distribution=self._weight_distribution(animals),
            image_count_distribution=self._image_count_distribution(animals),
            annotation_coverage=self._annotation_coverage(animals),
            invalid_animal_id_records=invalid_animal_id_records,
            records_with_missing_folder=self._count_records_with_missing_folder(record_list),
            records_with_missing_weight=self._count_records_with_missing_weight(record_list),
            records_with_missing_sex=self._count_records_with_missing_sex(record_list),
            animals_with_multiple_views=self._count_multiple_views(animals),
            animals_with_inconsistent_weight=self._count_inconsistent_weight(animals),
            animals_with_inconsistent_sex=self._count_inconsistent_sex(animals),
            inconsistent_weight_animals=[animal.animal_id for animal in animals if not animal.consistent_weight],
            inconsistent_sex_animals=[animal.animal_id for animal in animals if not animal.consistent_sex],
            animals=animals,
            metadata={
                "weight_tolerance": f"{self.weight_tolerance}",
                "dataset_membership_semantics": (
                    "Counts animals once per dataset membership. An animal present in B2 and B4 "
                    "contributes to both B2 and B4. This is not an exclusive partition."
                ),
                "metric_scope": (
                    "total_images and valid_image_count are image-level metrics. "
                    "All distributions and animal lists are animal-level metrics."
                ),
                "invalid_animal_id_behavior": (
                    "Records with missing or invalid animal_id are excluded from per-animal "
                    "analysis and counted separately in invalid_animal_id_records."
                ),
            },
        )

    def _resolve_records(self, records: Iterable[ImageRecord] | HasRecordsProtocol) -> list[ImageRecord]:
        if hasattr(records, "records"):
            return list(records.records)

        return list(records)

    def _count_invalid_animal_id_records(self, records: list[ImageRecord]) -> int:
        return sum(self._extract_animal_id(record) is None for record in records)

    def _group_by_animal(self, records: Iterable[ImageRecord]) -> tuple[dict[str, list[ImageRecord]], int]:
        groups: dict[str, list[ImageRecord]] = defaultdict(list)
        invalid_count = 0

        for record in records:
            animal_id = self._extract_animal_id(record)

            if animal_id is None:
                invalid_count += 1
                continue

            groups[animal_id].append(record)

        return {animal_id: sorted(group, key=lambda record: record.filename) for animal_id, group in groups.items()}, invalid_count

    def _analyze_animal(self, animal_id: str, records: list[ImageRecord]) -> AnimalRecordAnalysis:
        view_types = self._unique_sorted_views(records)
        dataset_types = self._unique_sorted_datasets(records)
        weight_values = [self._extract_weight(record) for record in records]
        distinct_weight_values = self._unique_sorted_weights(weight_values)
        consistent_weight = self._is_consistent_weight(weight_values)
        sex_values = [self._extract_sex(record) for record in records]
        distinct_sex_values = self._unique_sorted_sexes(sex_values)
        consistent_sex = self._is_consistent_sex(sex_values)
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

    def _extract_animal_id(self, record: ImageRecord) -> str | None:
        animal_id = record.animal_id

        if animal_id is None:
            return None

        normalized = str(animal_id).strip()

        if normalized == "":
            return None

        return normalized

    def _extract_dataset_type(self, record: ImageRecord) -> str:
        if record.folder is None:
            return "Unknown"

        try:
            return record.folder.dataset.value
        except Exception:
            return "Unknown"

    def _extract_view_type(self, record: ImageRecord) -> str:
        if record.folder is None:
            return "Unknown"

        try:
            return record.folder.view.value
        except Exception:
            return "Unknown"

    def _extract_weight(self, record: ImageRecord) -> float | None:
        weight = record.weight_kg

        if weight is None:
            return None

        if not isinstance(weight, (int, float)):
            return None

        if not math.isfinite(weight):
            return None

        return float(weight)

    def _extract_sex(self, record: ImageRecord) -> str:
        sex = record.sex

        if sex is None:
            return "Missing"

        if hasattr(sex, "value"):
            return str(sex.value)

        if isinstance(sex, str) and sex.strip() in {"M", "F"}:
            return sex.strip()

        return "Missing"

    def _unique_sorted_views(self, records: list[ImageRecord]) -> list[str]:
        return sorted({self._extract_view_type(record) for record in records})

    def _unique_sorted_datasets(self, records: list[ImageRecord]) -> list[str]:
        return sorted({self._extract_dataset_type(record) for record in records})

    def _unique_sorted_weights(self, weights: list[float | None]) -> list[float | None]:
        return sorted(set(weights), key=lambda value: (value is None, value))

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

    def _is_consistent_weight(self, weights: list[float | None]) -> bool:
        non_missing = [weight for weight in weights if weight is not None]

        if len(non_missing) == 0:
            return False

        if len(non_missing) != len(weights):
            return False

        return max(non_missing) - min(non_missing) <= self.weight_tolerance

    def _is_consistent_sex(self, sexes: list[str]) -> bool:
        distinct = set(sexes)

        return len(distinct) == 1 and next(iter(distinct)) in {"M", "F"}

    def _sex_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            if animal.consistent_sex and animal.distinct_sex_values:
                if animal.distinct_sex_values[0] in {"F", "M"}:
                    counter[animal.distinct_sex_values[0]] += 1
                    continue

            if animal.distinct_sex_values == ["Missing"]:
                counter["missing"] += 1
            else:
                counter["inconsistent"] += 1

        return dict(counter)

    def _dataset_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            for dataset_type in animal.dataset_types:
                counter[dataset_type] += 1

        return dict(counter)

    def _dataset_exclusive_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            dataset_types = [dataset for dataset in animal.dataset_types if dataset != "Unknown"]

            if not dataset_types:
                counter["Unknown"] += 1
                continue

            if len(dataset_types) == 1:
                counter[f"{dataset_types[0]} only"] += 1
                continue

            counter[" + ".join(dataset_types)] += 1

        return dict(counter)

    def _view_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter(animal.view_category for animal in animals)
        return dict(counter)

    def _weight_distribution(self, animals: list[AnimalRecordAnalysis]) -> dict[str, int]:
        counter = Counter()

        for animal in animals:
            if all(weight is None for weight in animal.weight_values):
                counter["missing"] += 1
                continue

            if not animal.consistent_weight:
                counter["inconsistent"] += 1
                continue

            weight_value = animal.distinct_weight_values[0]
            if weight_value is None:
                counter["missing"] += 1
                continue

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

    def _count_records_with_missing_folder(self, records: list[ImageRecord]) -> int:
        return sum(record.folder is None for record in records)

    def _count_records_with_missing_weight(self, records: list[ImageRecord]) -> int:
        return sum(self._extract_weight(record) is None for record in records)

    def _count_records_with_missing_sex(self, records: list[ImageRecord]) -> int:
        return sum(self._extract_sex(record) == "Missing" for record in records)

    def _median_image_count(self, animals: list[AnimalRecordAnalysis]) -> float:
        if not animals:
            return 0.0

        counts = sorted(animal.image_count for animal in animals)
        length = len(counts)
        middle = length // 2

        if length % 2 == 1:
            return float(counts[middle])

        return float((counts[middle - 1] + counts[middle]) / 2)

    def _validate_weight_tolerance(self, weight_tolerance: float) -> None:
        if not isinstance(weight_tolerance, (int, float)):
            raise TypeError("weight_tolerance must be numeric")

        if weight_tolerance < 0.0:
            raise ValueError("weight_tolerance must be non-negative")

    def _validate_weight_bins(self, weight_bins: list[float]) -> None:
        if len(weight_bins) < 2:
            raise ValueError("weight_bins must contain at least two values")

        if any(not isinstance(value, (int, float)) for value in weight_bins):
            raise TypeError("weight_bins values must be numeric")

        if any(weight_bins[i] >= weight_bins[i + 1] for i in range(len(weight_bins) - 1)):
            raise ValueError("weight_bins must be strictly increasing")

    def _bin_weight(self, weight: float) -> str:
        if weight < self.weight_bins[0]:
            return "below_minimum"

        for lower, upper in zip(self.weight_bins, self.weight_bins[1:]):
            if lower <= weight < upper:
                return f"{int(lower)}-{int(upper)}"

        return f"{int(self.weight_bins[-1])}+"
