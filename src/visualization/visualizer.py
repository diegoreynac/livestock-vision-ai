"""
Dataset visualizer.
"""

from __future__ import annotations

from pathlib import Path

import cv2

from src.core.context import ProjectContext

from src.dataset.models import ImageRecord
import numpy as np
from numpy.typing import NDArray
from src.visualization.constants import (
    BBOX_COLOR,
    SEGMENTATION_COLOR,
    KEYPOINT_COLOR,
    LABEL_COLOR,
    LINE_THICKNESS,
    POINT_RADIUS,
    FONT_SCALE,
    FONT_THICKNESS,
)


class DatasetVisualizer:
    """
    Visualizes dataset images and annotations.
    """

    def __init__(
        self,
        context: ProjectContext,
    ) -> None:

        self.context = context

        self.config = context.config

        self.logger = context.logger

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def visualize(
        self,
        record: ImageRecord,
        output_path: Path,
    ) -> Path:
        """
        Generate a visualization for one image.
        """

        image = self._load_image(record)

        canvas = image.copy()

        self._draw_bbox(
            canvas,
            record,
        )

        self._save_image(
            canvas,
            output_path,
        )

        return output_path

    # --------------------------------------------------
    # Image I/O
    # --------------------------------------------------

    def _load_image(
        self,
        record: ImageRecord,
    ) -> NDArray[np.uint8]:
        """
        Load image from disk.
        """

        image = cv2.imread(
            str(record.filepath),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            raise FileNotFoundError(
                f"Unable to load image: {record.filepath}"
            )

        return image

    def _save_image(
        self,
        image: NDArray[np.uint8],
        output_path: Path,
    ) -> None:
        """
        Save visualization image.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        success = cv2.imwrite(
            str(output_path),
            image,
        )

        if not success:

            raise RuntimeError(
                f"Unable to save image: {output_path}"
            )

        self.logger.info(
            f"Visualization saved: {output_path}"
        )

    def _draw_bbox(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """
        Draw annotation bounding box.
        """

        annotation = record.annotation

        if annotation is None:
            return

        bbox = annotation.bbox

        start_point = (
            int(bbox.x),
            int(bbox.y),
        )

        end_point = (
            int(bbox.right),
            int(bbox.bottom),
        )

        cv2.rectangle(
            image,
            start_point,
            end_point,
            color=BBOX_COLOR,
            thickness=2,
        )

    def _draw_segmentation(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """
        Draw segmentation polygons.
        """

        annotation = record.annotation

        if annotation is None:
            return

        for polygon in annotation.segmentation:

            points = np.array(
                polygon,
                dtype=np.int32,
            ).reshape(-1, 2)

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
        """
        Draw annotation keypoints.
        """

        annotation = record.annotation

        if annotation is None:
            return

        if annotation.keypoints is None:
            return

        for keypoint in annotation.keypoints:

            cv2.circle(
                image,
                (
                    int(keypoint.x),
                    int(keypoint.y),
                ),
                POINT_RADIUS,
                KEYPOINT_COLOR,
                -1,
            )

    def _draw_labels(
        self,
        image: NDArray[np.uint8],
        record: ImageRecord,
    ) -> None:
        """
        Draw image label.
        """

        annotation = record.annotation

        if annotation is None:
            return

        text = record.filename

        cv2.putText(
            image,
            text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            FONT_SCALE,
            LABEL_COLOR,
            FONT_THICKNESS,
            cv2.LINE_AA,
        )

    