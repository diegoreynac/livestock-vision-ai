from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


__all__ = [
    "BoundingBox",
    "clip_bbox",
    "coco_to_xyxy",
    "normalize_bbox",
    "validate_bbox",
    "validate_image_dimensions",
]


def validate_image_dimensions(image_width: int, image_height: int) -> tuple[int, int]:
    """Validate raw image dimensions and return normalized integer values."""
    if isinstance(image_width, bool) or isinstance(image_height, bool):
        raise ValueError("Image dimensions must be positive integers.")

    try:
        width_float = float(image_width)
        height_float = float(image_height)
        width = int(image_width)
        height = int(image_height)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError("Image dimensions must be positive integers.") from exc

    if not math.isfinite(width_float) or not math.isfinite(height_float):
        raise ValueError("Image dimensions must be finite positive integers.")
    if width <= 0 or height <= 0:
        raise ValueError("Image dimensions must be greater than zero.")
    if width != width_float or height != height_float:
        raise ValueError("Image dimensions must be integers.")
    return width, height


def _coerce_float(value: object, name: str) -> float:
    if value is None or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite numeric value.")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - defensive branch
        raise ValueError(f"{name} must be a finite numeric value.") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} must be a finite numeric value.")
    return number


def validate_bbox(
    x1: float,
    y1: float,
    x2: float,
    y2: float,
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> tuple[float, float, float, float]:
    """Validate a bounding box in XYXY form and return it as floats."""
    x1_val = _coerce_float(x1, "x1")
    y1_val = _coerce_float(y1, "y1")
    x2_val = _coerce_float(x2, "x2")
    y2_val = _coerce_float(y2, "y2")

    if x1_val < 0.0 or y1_val < 0.0 or x2_val < 0.0 or y2_val < 0.0:
        raise ValueError("Bounding box coordinates must be non-negative.")
    if x2_val <= x1_val:
        raise ValueError("Bounding box requires x2 > x1.")
    if y2_val <= y1_val:
        raise ValueError("Bounding box requires y2 > y1.")

    if image_width is not None or image_height is not None:
        if image_width is None or image_height is None:
            raise ValueError("Both image width and image height must be provided together.")
        width, height = validate_image_dimensions(image_width, image_height)
        if x2_val > width or y2_val > height:
            raise ValueError("Bounding box exceeds the image boundaries.")

    return x1_val, y1_val, x2_val, y2_val


@dataclass(frozen=True, slots=True)
class BoundingBox:
    """Normalized bounding box representation in XYXY coordinates."""

    x1: float
    y1: float
    x2: float
    y2: float
    image_width: int | None = None
    image_height: int | None = None

    def __post_init__(self) -> None:
        x1, y1, x2, y2 = validate_bbox(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            image_width=self.image_width,
            image_height=self.image_height,
        )
        object.__setattr__(self, "x1", x1)
        object.__setattr__(self, "y1", y1)
        object.__setattr__(self, "x2", x2)
        object.__setattr__(self, "y2", y2)

    @classmethod
    def from_coco(
        cls,
        coco_bbox: Sequence[float],
        *,
        image_width: int | None = None,
        image_height: int | None = None,
    ) -> "BoundingBox":
        """Construct a valid XYXY box from a COCO-style [x, y, width, height] box."""
        if len(coco_bbox) != 4:
            raise ValueError("COCO bounding boxes must have exactly four values: [x, y, width, height].")

        x, y, width, height = coco_bbox
        x_val = _coerce_float(x, "x")
        y_val = _coerce_float(y, "y")
        width_val = _coerce_float(width, "width")
        height_val = _coerce_float(height, "height")

        if x_val < 0.0 or y_val < 0.0:
            raise ValueError("COCO bounding box origin must be non-negative.")
        if width_val <= 0.0 or height_val <= 0.0:
            raise ValueError("COCO bounding box width and height must be positive.")

        if image_width is not None or image_height is not None:
            if image_width is None or image_height is None:
                raise ValueError("Both image width and image height must be provided together.")
            width_pixels, height_pixels = validate_image_dimensions(image_width, image_height)
            if x_val + width_val > width_pixels or y_val + height_val > height_pixels:
                raise ValueError("COCO bounding box exceeds the image boundaries.")

        return cls(
            x_val,
            y_val,
            x_val + width_val,
            y_val + height_val,
            image_width=image_width,
            image_height=image_height,
        )

    @property
    def xyxy(self) -> list[float]:
        return [self.x1, self.y1, self.x2, self.y2]

    def clip(self, *, image_width: int | None = None, image_height: int | None = None) -> "BoundingBox":
        target_width = self.image_width if image_width is None else image_width
        target_height = self.image_height if image_height is None else image_height
        if target_width is None or target_height is None:
            raise ValueError("Image dimensions are required to clip bounding boxes.")
        width, height = validate_image_dimensions(target_width, target_height)
        clipped_x1 = min(max(self.x1, 0.0), float(width))
        clipped_y1 = min(max(self.y1, 0.0), float(height))
        clipped_x2 = min(max(self.x2, 0.0), float(width))
        clipped_y2 = min(max(self.y2, 0.0), float(height))
        return BoundingBox(
            clipped_x1,
            clipped_y1,
            clipped_x2,
            clipped_y2,
            image_width=width,
            image_height=height,
        )

    def normalized(self, *, image_width: int | None = None, image_height: int | None = None) -> list[float]:
        target_width = self.image_width if image_width is None else image_width
        target_height = self.image_height if image_height is None else image_height
        if target_width is None or target_height is None:
            raise ValueError("Image dimensions are required to normalize bounding boxes.")
        width, height = validate_image_dimensions(target_width, target_height)
        validate_bbox(
            self.x1,
            self.y1,
            self.x2,
            self.y2,
            image_width=width,
            image_height=height,
        )
        return [self.x1 / width, self.y1 / height, self.x2 / width, self.y2 / height]


def coco_to_xyxy(
    coco_bbox: Sequence[float],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> BoundingBox:
    """Convert a COCO bbox [x, y, width, height] into a validated XYXY BoundingBox."""
    return BoundingBox.from_coco(coco_bbox, image_width=image_width, image_height=image_height)


def clip_bbox(
    bbox: BoundingBox | Sequence[float],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> BoundingBox:
    """Clip a bounding box to the image boundaries and return a valid BoundingBox."""
    if isinstance(bbox, BoundingBox):
        return bbox.clip(image_width=image_width, image_height=image_height)

    if len(bbox) != 4:
        raise ValueError("Bounding boxes must contain exactly four values: [x1, y1, x2, y2].")

    if image_width is None or image_height is None:
        raise ValueError("Image dimensions are required to clip bounding boxes.")

    width, height = validate_image_dimensions(image_width, image_height)
    x1, y1, x2, y2 = validate_bbox(*bbox)
    clipped_x1 = min(max(x1, 0.0), float(width))
    clipped_y1 = min(max(y1, 0.0), float(height))
    clipped_x2 = min(max(x2, 0.0), float(width))
    clipped_y2 = min(max(y2, 0.0), float(height))
    return BoundingBox(clipped_x1, clipped_y1, clipped_x2, clipped_y2, image_width=width, image_height=height)


def normalize_bbox(
    bbox: BoundingBox | Sequence[float],
    *,
    image_width: int | None = None,
    image_height: int | None = None,
) -> list[float]:
    """Normalize an XYXY box to the image-relative range [0, 1]."""
    if isinstance(bbox, BoundingBox):
        return bbox.normalized(image_width=image_width, image_height=image_height)

    if len(bbox) != 4:
        raise ValueError("Bounding boxes must contain exactly four values: [x1, y1, x2, y2].")

    if image_width is None or image_height is None:
        raise ValueError("Image dimensions are required to normalize bounding boxes.")

    box = BoundingBox(*bbox, image_width=image_width, image_height=image_height)
    return box.normalized(image_width=image_width, image_height=image_height)
