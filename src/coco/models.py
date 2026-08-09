"""
COCO data models.

Defines the object-oriented representation of COCO annotations.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.coco.enums import KeypointVisibility

from pathlib import Path


# ==========================================================
# Bounding Box
# ==========================================================

@dataclass(slots=True, frozen=True)
class COCOBoundingBox:
    """
    Bounding box in COCO format.
    """

    x: float
    y: float

    width: float
    height: float

    @property
    def area(self) -> float:
        """
        Bounding box area.
        """
        return self.width * self.height

    @property
    def center(self) -> tuple[float, float]:
        """
        Bounding box center.
        """
        return (
            self.x + self.width / 2,
            self.y + self.height / 2
        )

    @property
    def aspect_ratio(self) -> float:
        """
        Width / Height ratio.
        """

        if self.height == 0:
            return 0.0

        return self.width / self.height

# ==========================================================
# Keypoint
# ==========================================================

@dataclass(slots=True, frozen=True)
class COCOKeypoint:
    """
    Single COCO keypoint.
    """

    keypoint_id: int

    x: float

    y: float

    visibility: KeypointVisibility

    @property
    def is_visible(self) -> bool:
        """
        Returns True if the keypoint is visible.
        """

        return self.visibility.is_visible

    @property
    def is_labeled(self) -> bool:
        """
        Returns True if the keypoint is labeled.
        """

        return self.visibility.is_labeled
        
# ==========================================================
# Image
# ==========================================================

@dataclass(slots=True, frozen=True)
class COCOImage:
    """
    COCO image information.
    """

    id: int

    file_name: str

    width: int

    height: int

    @property
    def resolution(self) -> tuple[int, int]:
        """
        Image resolution.
        """

        return (
            self.width,
            self.height
        )

    @property
    def aspect_ratio(self) -> float:
        """
        Image aspect ratio.
        """

        if self.height == 0:
            return 0.0

        return self.width / self.height

    @property
    def pixel_count(self) -> int:
        """
        Total number of pixels.
        """

        return self.width * self.height
    
# ==========================================================
# Category
# ==========================================================

@dataclass(slots=True, frozen=True)
class COCOCategory:
    """
    COCO category information.
    """

    id: int

    name: str

    supercategory: str | None = None

    keypoints: list[str] = field(
        default_factory=list
    )

# ==========================================================
# Annotation
# ==========================================================

@dataclass(slots=True, frozen=True)
class COCOAnnotation:

    id: int

    image_id: int

    category_id: int

    bbox: COCOBoundingBox

    segmentation: list[list[float]] = field(
        default_factory=list
    )

    keypoints: list[COCOKeypoint] = field(
        default_factory=list
    )

    area: float = 0.0

    iscrowd: bool = False

    @property
    def has_keypoints(self) -> bool:
        """
        Returns True if keypoints exist.
        """

        return len(self.keypoints) > 0

    @property
    def visible_keypoints(self) -> list[COCOKeypoint]:
        """
        Returns only visible keypoints.
        """

        return [

            keypoint

            for keypoint in self.keypoints

            if keypoint.is_visible

        ]

    @property
    def labeled_keypoints(self) -> list[COCOKeypoint]:
        """
        Returns labeled keypoints.
        """

        return [

            keypoint

            for keypoint in self.keypoints

            if keypoint.is_labeled

        ]

    @property
    def visible_keypoint_count(self) -> int:
        """
        Number of visible keypoints.
        """

        return len(
            self.visible_keypoints
        )

    @property
    def labeled_keypoint_count(self) -> int:
        """
        Number of labeled keypoints.
        """

        return len(
            self.labeled_keypoints
        )

    def get_keypoint(
        self,
        keypoint_id: int
    ) -> COCOKeypoint | None:
        """
        Returns a keypoint by its identifier.
        """

        for keypoint in self.keypoints:

            if keypoint.keypoint_id == keypoint_id:

                return keypoint

        return None
    
