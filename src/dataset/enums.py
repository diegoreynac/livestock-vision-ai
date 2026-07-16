"""
Dataset enumerations.

Defines all enumerations used by the dataset module.
"""

from __future__ import annotations

from enum import Enum
from pathlib import Path


# ==========================================================
# DATASET TYPES
# ==========================================================

class DatasetType(Enum):
    """
    Supported dataset versions.
    """

    B2 = "B2"
    B3 = "B3"
    B4 = "B4"

    @classmethod
    def from_string(cls, value: str) -> "DatasetType":

        value = value.upper()

        try:
            return cls(value)

        except ValueError:

            raise ValueError(
                f"Unsupported dataset: {value}"
            )

    def __str__(self) -> str:

        return self.value


# ==========================================================
# CAMERA VIEW
# ==========================================================

class View(Enum):
    """
    Camera view.
    """

    SIDE = "Side"

    REAR = "Rear"

    @classmethod
    def from_string(cls, value: str) -> "View":

        value = value.lower()

        mapping = {
            "s": cls.SIDE,
            "side": cls.SIDE,
            "side_2": cls.SIDE,

            "r": cls.REAR,
            "rear": cls.REAR,
            "rear_2": cls.REAR,
            "back": cls.REAR,
        }

        if value not in mapping:

            raise ValueError(
                f"Unknown view: {value}"
            )

        return mapping[value]

    def __str__(self) -> str:

        return self.value


# ==========================================================
# SEX
# ==========================================================

class Sex(Enum):
    """
    Animal sex.
    """

    FEMALE = "F"

    MALE = "M"

    @classmethod
    def from_string(cls, value: str) -> "Sex":

        value = value.upper()

        try:

            return cls(value)

        except ValueError:

            raise ValueError(
                f"Unknown sex: {value}"
            )

    def __str__(self) -> str:

        return self.value


# ==========================================================
# IMAGE EXTENSIONS
# ==========================================================

class ImageExtension(Enum):
    """
    Supported image extensions.
    """

    JPG = ".jpg"

    JPEG = ".jpeg"

    PNG = ".png"

    @classmethod
    def is_supported(cls, path: Path) -> bool:
        """
        Returns True if the file extension is supported.
        """

        return path.suffix.lower() in {

            extension.value

            for extension in cls

        }

    def __str__(self) -> str:

        return self.value