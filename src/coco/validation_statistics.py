"""
Statistics produced by COCO validation.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class COCOValidationStatistics:
    """
    Statistics collected while validating
    COCO annotations.
    """

    images_checked: int = 0

    annotations_checked: int = 0

    missing_annotations: int = 0

    bbox_errors: int = 0

    segmentation_errors: int = 0

    keypoint_errors: int = 0

    warnings: int = 0

    errors: int = 0

    def reset(self) -> None:

        self.images_checked = 0
        self.annotations_checked = 0
        self.missing_annotations = 0
        self.bbox_errors = 0
        self.segmentation_errors = 0
        self.keypoint_errors = 0
        self.warnings = 0
        self.errors = 0

    @property
    def is_valid(self) -> bool:
        """
        Returns True if no validation errors were found.
        """

        return (
            self.errors == 0
            and self.bbox_errors == 0
            and self.segmentation_errors == 0
            and self.keypoint_errors == 0
            and self.missing_annotations == 0
        )

    def __str__(self) -> str:

        return (
            "\n"
            "COCO Validation Statistics\n"
            "------------------------------\n"
            f"Images checked      : {self.images_checked}\n"
            f"Annotations checked : {self.annotations_checked}\n"
            f"Missing annotations : {self.missing_annotations}\n"
            f"BBox errors         : {self.bbox_errors}\n"
            f"Segmentation errors : {self.segmentation_errors}\n"
            f"Keypoint errors     : {self.keypoint_errors}\n"
            f"Warnings            : {self.warnings}\n"
            f"Errors              : {self.errors}"
        )