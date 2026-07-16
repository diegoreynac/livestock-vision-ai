"""
Filename parser.

Parses image filenames into ImageRecord objects.
"""

from __future__ import annotations

from pathlib import Path

from src.dataset.enums import DatasetType
from src.dataset.enums import Sex

from src.dataset.models import (
    ImageFolder,
    ImageRecord
)


class FilenameParser:
    """
    Parses livestock image filenames.

    Supports:
        • B2
        • B3
        • B4
    """

    # =====================================================
    # PUBLIC
    # =====================================================

    def parse(
        self,
        folder: ImageFolder,
        filepath: Path
    ) -> ImageRecord:

        if folder.dataset == DatasetType.B2:

            return self._parse_b2(
                folder,
                filepath
            )

        if folder.dataset == DatasetType.B3:

            return self._parse_b2(
                folder,
                filepath
            )

        if folder.dataset == DatasetType.B4:

            return self._parse_b4(
                folder,
                filepath
            )

        raise ValueError(
            f"Unsupported dataset: {folder.dataset}"
        )

    # =====================================================
    # PRIVATE
    # =====================================================

    def _parse_b2(
        self,
        folder: ImageFolder,
        filepath: Path
    ) -> ImageRecord:

        parts = filepath.stem.split("_")

        if len(parts) != 5:

            raise ValueError(
                f"Invalid B2 filename: {filepath.name}"
            )

        animal_id = str(int(float(parts[0])))

        weight = float(parts[2])

        extra = float(parts[3])

        sex = Sex.from_string(parts[4])

        return ImageRecord(

            animal_id=animal_id,

            weight_kg=weight,

            sex=sex,

            filename=filepath.name,

            filepath=filepath,

            extra=extra

        )

    # -----------------------------------------------------

    def _parse_b4(
        self,
        folder: ImageFolder,
        filepath: Path
    ) -> ImageRecord:

        parts = filepath.stem.split("_")

        if len(parts) != 5:

            raise ValueError(
                f"Invalid B4 filename: {filepath.name}"
            )

        animal_id = f"{parts[0]}_{parts[1]}"

        weight = float(parts[3])

        sex = Sex.from_string(parts[4])

        return ImageRecord(

            animal_id=animal_id,

            weight_kg=weight,

            sex=sex,

            filename=filepath.name,

            filepath=filepath

        )