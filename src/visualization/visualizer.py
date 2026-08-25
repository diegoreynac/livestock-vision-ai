"""COCO annotation visualisation for livestock image records."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np
from numpy.typing import NDArray

from src.coco.enums import KeypointVisibility
from src.core.context import ProjectContext
from src.dataset.models import ImageRecord
from src.visualization.constants import (
    BBOX_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
    KEYPOINT_COLOR,
    LABEL_COLOR,
    LINE_THICKNESS,
    POINT_RADIUS,
    SEGMENTATION_COLOR,
)


SkeletonDefinition = Mapping[int, Sequence[tuple[int, int]]]


class DatasetVisualizer:
    """Render every available COCO annotation component on an image.

    ``skeletons`` is optional because the current ``COCOAnnotation`` model
    stores only ``category_id``.  When supplied, it maps a category id to
    COCO-standard one-based keypoint index pairs, for example ``{1: [(1, 2)]}``.
    Missing or malformed optional annotation fields are ignored safely.
    """

    _VISIBLE_KEYPOINT_COLOR = (0, 255, 0)
    _OCCLUDED_KEYPOINT_COLOR = (0, 165, 255)
    _UNLABELED_KEYPOINT_COLOR = (128, 128, 128)
    _UNKNOWN_KEYPOINT_COLOR = KEYPOINT_COLOR
    _SKELETON_COLOR = (255, 255, 0)

    def __init__(
        self,
        context: ProjectContext,
        skeletons: SkeletonDefinition | None = None,
    ) -> None:
        self.context = context
        self.config = context.config
        self.logger = context.logger
        self.skeletons = dict(skeletons or {})

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def visualize(
        self,
        record: ImageRecord,
        output_path: Path,
    ) -> Path:
        """Generate a visualisation without requiring all COCO fields."""

        image = self._load_image(record)
        canvas = image.copy()

        self._draw_segmentation(canvas, record)
        self._draw_bbox(canvas, record)
        self._draw_skeleton(canvas, record)
        self._draw_keypoints(canvas, record)
        self._draw_labels(canvas, record)

        self._save_image(canvas, output_path)
        return output_path

    # --------------------------------------------------
    # Image I/O
    # --------------------------------------------------

    def _load_image(self, record: ImageRecord) -> NDArray[np.uint8]:
        image = cv2.imread(str(record.filepath), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(f"Unable to load image: {record.filepath}")
        return image

    def _save_image(self, image: NDArray[np.uint8], output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(output_path), image):
            raise RuntimeError(f"Unable to save image: {output_path}")
        self.logger.info(f"Visualization saved: {output_path}")

    # --------------------------------------------------
    # COCO components
    # --------------------------------------------------

    def _draw_bbox(self, image: NDArray[np.uint8], record: ImageRecord) -> None:
        """Draw a COCO bbox using its ``[x, y, width, height]`` semantics."""

        annotation = record.annotation
        bbox = getattr(annotation, "bbox", None)
        if bbox is None:
            return

        values = (
            getattr(bbox, "x", None),
            getattr(bbox, "y", None),
            getattr(bbox, "width", None),
            getattr(bbox, "height", None),
        )
        if not self._finite_numbers(*values):
            return

        x, y, width, height = values
        if width <= 0 or height <= 0:
            return

        start = (int(x), int(y))
        end = (int(x + width), int(y + height))
        cv2.rectangle(image, start, end, BBOX_COLOR, LINE_THICKNESS)

    def _draw_segmentation(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """Draw each valid polygon in an optional COCO polygon segmentation."""

        annotation = record.annotation
        segmentation = getattr(annotation, "segmentation", None)
        if not isinstance(segmentation, (list, tuple)):
            return

        for polygon in segmentation:
            points = self._polygon_points(polygon)
            if points is None:
                continue
            cv2.polylines(
                image,
                [points],
                isClosed=True,
                color=SEGMENTATION_COLOR,
                thickness=LINE_THICKNESS,
            )

    def _draw_keypoints(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """Draw all available keypoints, independently of their cardinality."""

        annotation = record.annotation
        for keypoint in self._keypoints(annotation):
            point = self._keypoint_point(keypoint)
            if point is None:
                continue
            cv2.circle(
                image,
                point,
                POINT_RADIUS,
                self._keypoint_color(getattr(keypoint, "visibility", None)),
                -1,
            )

    def _draw_skeleton(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """Draw configured one-based COCO skeleton edges when both ends exist."""

        annotation = record.annotation
        category_id = getattr(annotation, "category_id", None)
        skeleton = self.skeletons.get(category_id)
        if not skeleton:
            return

        keypoints_by_id = {
            getattr(keypoint, "keypoint_id", None): keypoint
            for keypoint in self._keypoints(annotation)
        }
        for edge in skeleton:
            if not self._valid_skeleton_edge(edge):
                continue

            start_id, end_id = edge[0] - 1, edge[1] - 1
            start_keypoint = keypoints_by_id.get(start_id)
            end_keypoint = keypoints_by_id.get(end_id)
            if not self._is_labeled(start_keypoint) or not self._is_labeled(end_keypoint):
                continue

            start = self._keypoint_point(start_keypoint)
            end = self._keypoint_point(end_keypoint)
            if start is None or end is None:
                continue
            cv2.line(image, start, end, self._SKELETON_COLOR, LINE_THICKNESS)

    def _draw_labels(self, image: NDArray[np.uint8], record: ImageRecord) -> None:
        """Show category id, keypoint cardinality and annotated area."""

        annotation = record.annotation
        if annotation is None:
            return

        category_id = getattr(annotation, "category_id", "n/a")
        keypoint_count = len(self._keypoints(annotation))
        area = self._format_area(getattr(annotation, "area", None))
        text = f"cat={category_id} | kp={keypoint_count} | area={area}"

        (width, height), baseline = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            FONT_THICKNESS,
        )
        cv2.rectangle(image, (5, 5), (15 + width, 15 + height + baseline), (0, 0, 0), -1)
        cv2.putText(
            image,
            text,
            (10, 10 + height),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            LABEL_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    # --------------------------------------------------
    # Defensive conversion helpers
    # --------------------------------------------------

    @staticmethod
    def _finite_numbers(*values: object) -> bool:
        return all(
            isinstance(value, (int, float)) and math.isfinite(value)
            for value in values
        )

    def _polygon_points(self, polygon: object) -> NDArray[np.int32] | None:
        if not isinstance(polygon, (list, tuple)) or len(polygon) < 6:
            return None
        if len(polygon) % 2 != 0 or not self._finite_numbers(*polygon):
            return None

        try:
            return np.asarray(polygon, dtype=np.int32).reshape(-1, 2)
        except (TypeError, ValueError, OverflowError):
            return None

    @staticmethod
    def _keypoints(annotation: object) -> list[object]:
        keypoints = getattr(annotation, "keypoints", None)
        if not isinstance(keypoints, (list, tuple)):
            return []
        return list(keypoints)

    def _keypoint_point(self, keypoint: object) -> tuple[int, int] | None:
        if keypoint is None:
            return None
        x, y = getattr(keypoint, "x", None), getattr(keypoint, "y", None)
        if not self._finite_numbers(x, y):
            return None
        return int(x), int(y)

    @classmethod
    def _keypoint_color(cls, visibility: object) -> tuple[int, int, int]:
        try:
            visibility = KeypointVisibility(visibility)
        except (TypeError, ValueError):
            return cls._UNKNOWN_KEYPOINT_COLOR

        if visibility == KeypointVisibility.VISIBLE:
            return cls._VISIBLE_KEYPOINT_COLOR
        if visibility == KeypointVisibility.OCCLUDED:
            return cls._OCCLUDED_KEYPOINT_COLOR
        return cls._UNLABELED_KEYPOINT_COLOR

    @staticmethod
    def _valid_skeleton_edge(edge: object) -> bool:
        return (
            isinstance(edge, (tuple, list))
            and len(edge) == 2
            and all(isinstance(index, int) and index > 0 for index in edge)
        )

    @staticmethod
    def _is_labeled(keypoint: object) -> bool:
        if keypoint is None:
            return False
        try:
            return KeypointVisibility(getattr(keypoint, "visibility", None)).is_labeled
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _format_area(area: object) -> str:
        if isinstance(area, (int, float)) and math.isfinite(area):
            return f"{area:.1f}"
        return "n/a"
