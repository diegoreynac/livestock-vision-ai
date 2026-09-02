"""PyTorch dataset for canonical livestock training samples."""

from __future__ import annotations

from enum import Enum
import math
from collections.abc import Collection
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from src.training.augmentation import (
    AugmentationConfig,
    eval_preprocess,
    training_preprocess,
)
from src.training.bbox_preprocessing import BoundingBox
from src.training.samples import TrainingSample


class InputMode(str, Enum):
    """Image views exposed by :class:`LivestockDataset`."""

    SIDE = "side"
    REAR = "rear"
    SIDE_REAR = "side_rear"

    @classmethod
    def from_value(cls, value: "InputMode | str") -> "InputMode":
        if isinstance(value, cls):
            return value
        try:
            return cls(str(value).lower())
        except ValueError as exc:
            supported = ", ".join(mode.value for mode in cls)
            raise ValueError(
                f"Unsupported input mode {value!r}; expected one of: {supported}."
            ) from exc


class LivestockDataset(Dataset[dict[str, Any]]):
    """Lazily load and preprocess canonical ``TrainingSample`` objects."""

    def __init__(
        self,
        samples: Collection[TrainingSample],
        *,
        input_mode: InputMode | str = InputMode.SIDE,
        training: bool = True,
        augmentation_config: AugmentationConfig | None = None,
    ) -> None:
        if samples is None:
            raise ValueError("samples must be a collection of TrainingSample objects.")
        if isinstance(samples, (str, bytes)):
            raise TypeError("samples must be a collection of TrainingSample objects.")

        try:
            self._samples = tuple(samples)
        except TypeError as exc:
            raise TypeError("samples must be an iterable of TrainingSample objects.") from exc

        if any(not isinstance(sample, TrainingSample) for sample in self._samples):
            raise TypeError("Every dataset sample must be a TrainingSample instance.")

        self.input_mode = InputMode.from_value(input_mode)
        self.training = bool(training)
        self.augmentation_config = (
            augmentation_config if augmentation_config is not None else AugmentationConfig()
        )

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self._samples[index]
        paths = self._paths_for_sample(sample)
        annotations = self._annotations_for_sample(sample)

        # Separate RNGs with the same seed keep geometric pairing identical while
        # avoiding a second image search or a shared mutable annotation object.
        shared_seed = index
        images: list[torch.Tensor] = []
        processed_boxes: list[list[float]] = []
        for position, (path, annotation) in enumerate(zip(paths, annotations)):
            image = self._load_image(path)
            height, width = image.shape[:2]
            box = self._initial_bbox(annotation, width=width, height=height)
            spatial = {"boxes": np.asarray([box.xyxy], dtype=np.float32)}

            if self.training:
                processed = training_preprocess(
                    image,
                    self.augmentation_config,
                    seed=shared_seed,
                    annotations=spatial,
                )
            else:
                processed = eval_preprocess(
                    image,
                    self.augmentation_config,
                    annotations=spatial,
                )

            processed_height, processed_width = processed.shape[:2]
            clipped = self._clip_transformed_bbox(
                spatial["boxes"][0],
                width=processed_width,
                height=processed_height,
            )
            processed_boxes.append(clipped.normalized())
            images.append(self._to_tensor(processed))

        if self.input_mode is InputMode.SIDE_REAR:
            if images[0].shape[1:] != images[1].shape[1:]:
                raise ValueError(
                    "Side and Rear images have incompatible dimensions after preprocessing: "
                    f"{tuple(images[0].shape[1:])} and {tuple(images[1].shape[1:])}."
                )
            image_tensor = torch.cat(images, dim=0)
        else:
            image_tensor = images[0]

        return {
            "image": image_tensor,
            "bbox": torch.tensor(processed_boxes[0], dtype=torch.float32),
            "weight": torch.tensor([self._weight(sample)], dtype=torch.float32),
            "animal_id": sample.animal_id,
        }

    def _paths_for_sample(self, sample: TrainingSample) -> tuple[Path, ...]:
        if self.input_mode is InputMode.SIDE:
            return (self._required_path(sample.side_image, "Side", sample.animal_id),)
        if self.input_mode is InputMode.REAR:
            return (self._required_path(sample.rear_image, "Rear", sample.animal_id),)
        return (
            self._required_path(sample.side_image, "Side", sample.animal_id),
            self._required_path(sample.rear_image, "Rear", sample.animal_id),
        )

    def _annotations_for_sample(self, sample: TrainingSample) -> tuple[Any, ...]:
        if self.input_mode is InputMode.SIDE:
            return (self._required_annotation(sample.side_annotation, "Side", sample.animal_id),)
        if self.input_mode is InputMode.REAR:
            return (self._required_annotation(sample.rear_annotation, "Rear", sample.animal_id),)
        return (
            self._required_annotation(sample.side_annotation, "Side", sample.animal_id),
            self._required_annotation(sample.rear_annotation, "Rear", sample.animal_id),
        )

    @staticmethod
    def _required_path(path: Path | None, view: str, animal_id: str) -> Path:
        if path is None:
            raise ValueError(f"Missing {view} image for animal {animal_id!r}.")
        return Path(path)

    @staticmethod
    def _required_annotation(annotation: Any, view: str, animal_id: str) -> Any:
        if annotation is None or getattr(annotation, "bbox", None) is None:
            raise ValueError(f"Missing valid {view} COCO bounding box for animal {animal_id!r}.")
        return annotation

    @staticmethod
    def _load_image(path: Path) -> np.ndarray:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"Unable to read image for LivestockDataset: {path}")
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected a three-channel image at {path}, got shape {image.shape}.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if image.shape[0] <= 0 or image.shape[1] <= 0:
            raise ValueError(f"Image has invalid dimensions at {path}: {image.shape}.")
        return image

    @staticmethod
    def _initial_bbox(annotation: Any, *, width: int, height: int) -> BoundingBox:
        coco = annotation.bbox
        return BoundingBox.from_coco(
            (coco.x, coco.y, coco.width, coco.height),
            image_width=width,
            image_height=height,
        )

    @staticmethod
    def _clip_transformed_bbox(values: Any, *, width: int, height: int) -> BoundingBox:
        coordinates = np.asarray(values, dtype=np.float32)
        if coordinates.shape != (4,) or not np.all(np.isfinite(coordinates)):
            raise ValueError("Transformed bounding box must contain four finite coordinates.")
        x1, y1, x2, y2 = coordinates.tolist()
        clipped = (max(0.0, min(x1, width)), max(0.0, min(y1, height)),
                   max(0.0, min(x2, width)), max(0.0, min(y2, height)))
        if clipped[2] <= clipped[0] or clipped[3] <= clipped[1]:
            raise ValueError("Transformed bounding box is empty after clipping.")
        return BoundingBox(*clipped, image_width=width, image_height=height)

    @staticmethod
    def _to_tensor(image: np.ndarray) -> torch.Tensor:
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected HWC RGB image, got shape {image.shape}.")
        return torch.from_numpy(np.ascontiguousarray(image, dtype=np.float32)).permute(2, 0, 1)

    @staticmethod
    def _weight(sample: TrainingSample) -> float:
        value = sample.weight_kg
        if isinstance(value, bool):
            raise ValueError(f"Invalid weight for animal {sample.animal_id!r}: {value!r}.")
        try:
            weight = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid weight for animal {sample.animal_id!r}: {value!r}."
            ) from exc
        if not math.isfinite(weight):
            raise ValueError(f"Invalid weight for animal {sample.animal_id!r}: {value!r}.")
        return weight


__all__ = ["InputMode", "LivestockDataset"]
