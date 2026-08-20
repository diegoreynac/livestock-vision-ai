from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelOutput:
    """Architecture-independent representation of model predictions.

    Fields:
    - bbox: bounding box prediction (framework-agnostic, e.g., tuple/list/array)
    - sex: predicted sex label (e.g., 'M'/'F' or numeric encoding)
    - weight: predicted weight value (e.g., float representing kg)

    This dataclass intentionally does not depend on any specific tensor type or
    deep-learning framework.
    """

    bbox: Any
    sex: Any
    weight: Any
