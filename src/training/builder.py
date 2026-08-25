from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable, Optional

from pathlib import Path

from src.dataset.models import ImageRecord
from src.dataset.enums import Sex
from src.training.samples import TrainingSample


@dataclass(slots=True)
class DatasetBuilder:
    """Builds dual-view TrainingSample objects from ImageRecord inputs.

    Strict thesis data-quality policy:
    - default: require both Side and Rear records for a TrainingSample
    - invalid animal_id values (None, empty, whitespace-only) are excluded
    - missing sex or weight raises ValueError
    - inconsistent sex or weight raises ValueError

    When multiple records exist for the same view, the builder uses a
    deterministic policy: prefer annotated records, then order by filename.
    """

    require_both_views: bool = True

    def build(self, records: Iterable[ImageRecord]) -> list[TrainingSample]:
        groups = self._group_by_animal(records)
        samples: list[TrainingSample] = []

        for animal_id, group in sorted(groups.items()):
            side_records = [record for record in group if record.is_side]
            rear_records = [record for record in group if record.is_rear]

            if self.require_both_views and (not side_records or not rear_records):
                continue

            side_choice = self._choose_preferred(side_records)
            rear_choice = self._choose_preferred(rear_records)

            if side_choice is None and rear_choice is None:
                continue

            sex = self._resolve_sex(group)
            weight = self._resolve_weight(group)

            sample = TrainingSample(
                animal_id=animal_id,
                side_image=Path(side_choice.filepath) if side_choice is not None else None,
                rear_image=Path(rear_choice.filepath) if rear_choice is not None else None,
                side_annotation=side_choice.annotation if side_choice is not None else None,
                rear_annotation=rear_choice.annotation if rear_choice is not None else None,
                sex=sex,
                weight_kg=weight,
            )

            samples.append(sample)

        return samples

    def _group_by_animal(self, records: Iterable[ImageRecord]) -> dict[str, list[ImageRecord]]:
        groups: dict[str, list[ImageRecord]] = defaultdict(list)

        for record in records:
            animal_id = self._extract_animal_id(record)

            if animal_id is None:
                continue

            groups[animal_id].append(record)

        return {animal_id: sorted(group, key=lambda record: record.filename) for animal_id, group in groups.items()}

    def _extract_animal_id(self, record: ImageRecord) -> Optional[str]:
        animal_id = record.animal_id

        if animal_id is None:
            return None

        normalized = str(animal_id).strip()

        if normalized == "":
            return None

        return normalized

    def _choose_preferred(self, records: list[ImageRecord]) -> Optional[ImageRecord]:
        if not records:
            return None

        return sorted(records, key=lambda record: (record.annotation is None, record.filename))[0]

    def _resolve_sex(self, records: list[ImageRecord]) -> Sex:
        observed: list[str] = []

        for record in records:
            sex = record.sex

            if sex is None:
                continue

            if hasattr(sex, "value"):
                observed.append(str(sex.value).upper())
            else:
                observed.append(str(sex).upper())

        if not observed:
            raise ValueError("Missing sex metadata for animal; sex labels must be supplied upstream.")

        distinct = sorted(set(observed))
        if len(distinct) != 1:
            raise ValueError(f"Inconsistent sex values for animal: {distinct}")

        return Sex.from_string(distinct[0])

    def _resolve_weight(self, records: list[ImageRecord]) -> float:
        weights: list[float] = []

        for record in records:
            value = record.weight_kg

            if value is None:
                continue

            if isinstance(value, (int, float)):
                weights.append(float(value))

        if not weights:
            raise ValueError("Missing weight metadata for animal; weight values must be supplied upstream.")

        distinct = sorted({float(weight) for weight in weights})
        if len(distinct) != 1:
            raise ValueError(f"Inconsistent weight values for animal: {distinct}")

        return float(distinct[0])
