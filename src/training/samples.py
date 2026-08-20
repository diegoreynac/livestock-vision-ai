from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.coco.models import COCOAnnotation
from src.dataset.enums import Sex


@dataclass(slots=True)
class TrainingSample:
    animal_id: str
    side_image: Path | None
    rear_image: Path | None
    side_annotation: COCOAnnotation | None
    rear_annotation: COCOAnnotation | None
    sex: Sex
    weight_kg: float
