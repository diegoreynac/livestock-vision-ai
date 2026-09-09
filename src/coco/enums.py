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


# ==========================================================
# Annotation Status (audit)
# ==========================================================

class AnnotationStatus(Enum):
    """
    Per-image/per-annotation bounding box quality classification, used by
    the read-only annotation quality audit.
    """

    # Annotation exists, bbox has four finite values, positive width and
    # height, and lies within the image boundaries.
    VALID = "valid"

    # The image exists in COCO images[] but no annotation references it.
    MISSING = "missing"

    # An annotation exists and its bbox is exactly [0, 0, 0, 0].
    ZERO_BBOX = "zero_bbox"

    # An annotation exists but its bbox is structurally invalid: wrong
    # shape, non-finite values, or non-positive width/height.
    INVALID_BBOX = "invalid_bbox"

    # BBox is numerically valid with positive dimensions, but extends
    # outside the image boundaries.
    OUT_OF_BOUNDS = "out_of_bounds"

    # More than one annotation references the same image. This is an
    # audit finding, not automatically an error.
    MULTIPLE_ANNOTATIONS = "multiple_annotations"

    def __str__(self) -> str:

        return self.value
