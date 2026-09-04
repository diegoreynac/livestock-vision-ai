from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class ModelOutput:
    """Architecture-independent representation of model predictions.

    Fields:
    - bbox_side: bounding box prediction in the side-view image coordinate
      system (e.g., a (B, 4) tensor from forward() or a tuple/list of floats
      from predict()). Optional: populated when the side view is used.
    - bbox_rear: bounding box prediction in the rear-view image coordinate
      system. Optional: populated when the rear view is used.
    - weight: predicted weight value (e.g., a (B, 1) tensor from forward() or
      a float representing kg from predict()). Always a single regression
      output.
    - sex: predicted sex label or logits. Optional and kept temporarily for
      API compatibility; the thesis model does not train sex for now.
    - bbox: legacy single bounding box from the original dual-view contract.
      Deprecated: new code should use bbox_side / bbox_rear. Retained so
      existing models and callers keep working until they migrate to the
      per-view contract.

    Every field accepts both framework tensors (e.g., torch.Tensor produced by
    forward()) and plain Python values (produced by predict()); this dataclass
    intentionally does not depend on any specific tensor type or deep-learning
    framework. InputMode-dependent validation belongs to the model, not to
    this container.
    """

    bbox_side: Any = None
    bbox_rear: Any = None
    weight: Any = None
    sex: Any = None
    bbox: Any = None
