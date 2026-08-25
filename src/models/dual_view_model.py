from __future__ import annotations

from pathlib import Path
from typing import Any

from src.models.base import BaseModel
from src.models.output import ModelOutput
from src.models.architectures import mobilenet, efficientnet, yolo, ArchitectureSpec


class DualViewModel(BaseModel):
    """Dual-view model wrapper that is framework-agnostic.

    Design notes:
    - The model is intentionally a thin wrapper around an ArchitectureSpec.
      This keeps the BaseModel interface stable and avoids coupling to a
    specific deep-learning framework (PyTorch/TensorFlow).
    - forward() accepts two inputs (side, rear). For the purposes of the
      requested "dummy forward" the inputs are not interpreted as tensors; the
      implementation returns a deterministic ModelOutput structure. Downstream
      integrations can replace the internals with real backbones when a
      framework is chosen.
    - Parameter counting and size estimation are delegated to ArchitectureSpec
      so that tests and tooling can work without heavy dependencies.
    """

    def __init__(self, architecture: str = "mobilenet", variant: str = "default") -> None:
        arch_map = {
            "mobilenet": mobilenet,
            "efficientnet": efficientnet,
            "yolo": yolo,
        }
        if architecture not in arch_map:
            raise ValueError(f"Unknown architecture: {architecture}")

        factory = arch_map[architecture]
        self._spec: ArchitectureSpec = factory(variant)

    def forward(self, side: Any, rear: Any, **kwargs: Any) -> ModelOutput:
        """Perform a dummy forward that combines side and rear views.

        The forward returns a ModelOutput with the expected fields:
        - bbox: a framework-agnostic bounding box representation (xmin, ymin, xmax, ymax)
        - sex: a predicted sex label (string)
        - weight: a predicted numeric weight

        Values are deterministic stubs useful for testing interfaces rather
        than representing a real inference pipeline.
        """
        # Deterministic placeholders so tests do not depend on randomness
        bbox = (0.0, 0.0, 1.0, 1.0)
        sex = "M"
        weight = float(self.count_parameters() % 500)  # small deterministic number
        return ModelOutput(bbox=bbox, sex=sex, weight=weight)

    def predict(self, side: Any, rear: Any, **kwargs: Any) -> ModelOutput:
        # For the dummy implementation, predict delegates to forward.
        return self.forward(side, rear, **kwargs)

    def count_parameters(self) -> int:
        return self._spec.count_parameters()

    def model_size(self) -> float:
        return float(self._spec.model_size_mb())

    def export(self, destination: Path | str, **kwargs: Any) -> None:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        metadata = (
            f"architecture={self._spec.name}\n"
            f"variant={self._spec.variant}\n"
            f"parameters={self.count_parameters()}\n"
        )
        (dest / "model_metadata.txt").write_text(metadata, encoding="utf-8")
