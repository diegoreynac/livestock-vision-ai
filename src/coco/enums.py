"""
COCO enumerations.

This module defines the enumerations used throughout the
COCO annotation analysis.
"""

from enum import Enum, IntEnum, auto


# ==========================================================
# Annotation Types
# ==========================================================

class AnnotationType(Enum):
    """
    Supported COCO annotation types.
    """

    BOUNDING_BOX = auto()

    SEGMENTATION = auto()

    KEYPOINTS = auto()


# ==========================================================
# Keypoint Visibility
# ==========================================================

class KeypointVisibility(IntEnum):
    """
    COCO keypoint visibility flags.

    Values follow the official COCO specification.

    0 -> Not labeled
    1 -> Labeled but not visible (occluded)
    2 -> Visible
    """

    NOT_LABELED = 0

    OCCLUDED = 1

    VISIBLE = 2

    @property
    def is_labeled(self) -> bool:
        """
        Returns True if the keypoint was labeled.
        """

        return self != KeypointVisibility.NOT_LABELED

    @property
    def is_visible(self) -> bool:
        """
        Returns True if the keypoint is visible.
        """

        return self == KeypointVisibility.VISIBLE
    
# ==========================================================
# Keypoint Names
# ==========================================================

class KeypointName(Enum):
    """
    Supported livestock keypoints.
    """

    HEAD = auto()

    WITHERS = auto()

    BACK = auto()

    TAIL_HEAD = auto()

    LEFT_FRONT_HOOF = auto()

    RIGHT_FRONT_HOOF = auto()

    LEFT_REAR_HOOF = auto()

    RIGHT_REAR_HOOF = auto()

