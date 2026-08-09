"""
COCO dataset validator.
"""

from __future__ import annotations

from src.coco.enums import KeypointVisibility

from src.coco.models import COCOAnnotation

from src.core.context import ProjectContext

from src.dataset.livestock_dataset import LivestockDataset
from src.dataset.models import ImageRecord

from src.coco.validation_statistics import (
    COCOValidationStatistics,
)


class COCOValidator:
    """
    Validates COCO annotations attached to a
    LivestockDataset.
    """

    _EXPECTED_KEYPOINTS = 17

    _EXPECTED_REAR_KEYPOINTS = 4

    _EXPECTED_SIDE_KEYPOINTS = 9

    def __init__(
        self,
        context: ProjectContext,
    ) -> None:
        
        self.context = context

        self.config = context.config

        self.logger = context.logger

        self.statistics = COCOValidationStatistics()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def validate(
        self,
        dataset: LivestockDataset,
    ) -> COCOValidationStatistics:
        """
        Validate all COCO annotations in the dataset.
        """

        self.statistics.reset()

        self.logger.section(
            "COCO Validation"
        )

        for record in dataset:

            self.statistics.images_checked += 1

            self._validate_annotation(record)

        self.logger.info(
            str(self.statistics)
        )

        return self.statistics

    # --------------------------------------------------
    # Validation
    # --------------------------------------------------

    def _validate_annotation(
        self,
        record: ImageRecord,
    ) -> None:
        """
        Validate one image annotation.
        """

        annotation = record.annotation

        if annotation is None:

            self.statistics.missing_annotations += 1

            self._warning(
                f"{record.filename}: Missing annotation."
            )

            return

        self.statistics.annotations_checked += 1

        self._validate_bbox(
            record,
            annotation,
        )

        self._validate_keypoints(
            record,
            annotation,
        )

    # --------------------------------------------------
    # Logging
    # --------------------------------------------------

    def _warning(
        self,
        message: str,
    ) -> None:

        self.statistics.warnings += 1

        self.logger.warning(message)

    def _error(
        self,
        message: str,
    ) -> None:

        self.statistics.errors += 1

        self.logger.error(message)

    def _validate_bbox(
        self,
        record: ImageRecord,
        annotation: COCOAnnotation,
    ) -> None:
        """
        Validate annotation bounding box.
        """

        bbox = annotation.bbox

        if bbox.width <= 0:

            self.statistics.bbox_errors += 1

            self._error(
                f"{record.filename}: Invalid bbox width ({bbox.width})."
            )

        if bbox.height <= 0:

            self.statistics.bbox_errors += 1

            self._error(
                f"{record.filename}: Invalid bbox height ({bbox.height})."
            )

        if bbox.x < 0:

            self.statistics.bbox_errors += 1

            self._error(
                f"{record.filename}: Invalid bbox x ({bbox.x})."
            )

        if bbox.y < 0:

            self.statistics.bbox_errors += 1

            self._error(
                f"{record.filename}: Invalid bbox y ({bbox.y})."
            )

        if bbox.area <= 0:

            self.statistics.bbox_errors += 1

            self._error(
                f"{record.filename}: Invalid bbox area ({bbox.area})."
            )

    def _validate_segmentation(
        self,
        record: ImageRecord,
        annotation,
    ) -> None:
        """
        Validate polygon segmentation.
        """

        segmentation = annotation.segmentation

        if not segmentation:

            self.statistics.segmentation_errors += 1

            self._error(
                f"{record.filename}: Missing segmentation."
            )

            return

        for polygon in segmentation:

            if len(polygon) < 6:

                self.statistics.segmentation_errors += 1

                self._error(
                    f"{record.filename}: Polygon has fewer than 3 points."
                )

                continue

            if len(polygon) % 2 != 0:

                self.statistics.segmentation_errors += 1

                self._error(
                    f"{record.filename}: Polygon contains an odd number of coordinates."
                )

                continue

            for value in polygon:

                if value < 0:

                    self.statistics.segmentation_errors += 1

                    self._error(
                        f"{record.filename}: Negative segmentation coordinate ({value})."
                    )

                    break

    def _validate_keypoints(
        self,
        record: ImageRecord,
        annotation: COCOAnnotation,
    ) -> None:
        """
        Validate annotation keypoints.
        """

        keypoints = annotation.keypoints

        if not keypoints:

            self.statistics.keypoint_errors += 1

            self._error(
                f"{record.filename}: Missing keypoints."
            )

            return

        for index, keypoint in enumerate(keypoints):

            if keypoint.x < 0:

                self.statistics.keypoint_errors += 1

                self._error(
                    f"{record.filename}: Negative x coordinate for keypoint #{index}."
                )

            if keypoint.y < 0:

                self.statistics.keypoint_errors += 1

                self._error(
                    f"{record.filename}: Negative y coordinate for keypoint #{index}."
                )

            if not isinstance(
                keypoint.visibility,
                KeypointVisibility,
            ):

                self.statistics.keypoint_errors += 1

                self._error(
                    f"{record.filename}: Invalid visibility for keypoint #{index}."
                )                        

     