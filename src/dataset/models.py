"""
Dataset models.

Defines the data structures used throughout the dataset module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from src.dataset.enums import DatasetType
from src.dataset.enums import Sex
from src.dataset.enums import View
from src.coco.models import (
    COCOAnnotation
)


# ==========================================================
# IMAGE RECORD
# ==========================================================

@dataclass(slots=True)
class ImageRecord:
    """
    Represents a single image in the dataset.
    """

    animal_id: str

    weight_kg: float

    sex: Sex

    filename: str

    filepath: Path

    extra: float | None = None

    folder: "ImageFolder | None" = None

    # -------------------------------
    # Stage 2 (COCO)
    # -------------------------------

    annotation: COCOAnnotation | None = None

    # ------------------------------------------------------

    @property
    def dataset(self) -> DatasetType:

        if self.folder is None:

            raise RuntimeError(
                "ImageRecord is not attached to an ImageFolder."
            )

        return self.folder.dataset

    @property
    def view(self) -> View:

        return self.folder.view

    @property
    def is_side(self) -> bool:

        return self.view == View.SIDE

    @property
    def is_rear(self) -> bool:

        return self.view == View.REAR

    @property
    def is_male(self) -> bool:

        return self.sex == Sex.MALE

    @property
    def is_female(self) -> bool:

        return self.sex == Sex.FEMALE

    @property
    def stem(self) -> str:

        return self.filepath.stem

    @property
    def extension(self) -> str:

        return self.filepath.suffix.lower()

    def to_dict(self) -> dict:

        return {

            "dataset": self.dataset.value,

            "folder": self.folder.folder_name,

            "view": self.view.value,

            "animal_id": self.animal_id,

            "weight_kg": self.weight_kg,

            "sex": self.sex.value,

            "extra": self.extra,

            "filename": self.filename,

            "filepath": str(self.filepath),


            "has_annotation": self.has_annotation

        }

    @property
    def has_annotation(self) -> bool:
        """
        Returns True if the image has a COCO annotation.
        """

        return self.annotation is not None

# ==========================================================
# IMAGE FOLDER
# ==========================================================

@dataclass(slots=True)
class ImageFolder:
    """
    Represents one image folder inside the dataset.
    """

    dataset: DatasetType

    folder_name: str

    view: View

    path: Path

    records: list[ImageRecord] = field(default_factory=list)

    # ------------------------------------------------------

    annotation_file: Path | None = None

    @property
    def has_coco(self) -> bool:
        return (
            self.annotation_file is not None
            and self.annotation_file.exists()
        )

    @property
    def image_count(self) -> int:

        return len(self.records)

    @property
    def animal_count(self) -> int:

        return len(
            {
                record.animal_id
                for record in self.records
            }
        )

    @property
    def male_count(self) -> int:

        return sum(
            record.is_male
            for record in self.records
        )

    @property
    def female_count(self) -> int:

        return sum(
            record.is_female
            for record in self.records
        )

    def __iter__(self):

        return iter(self.records)

    def __len__(self):

        return len(self.records)

    def __getitem__(self, index):

        return self.records[index]

    def __str__(self):

        return (
            f"{self.dataset.value}"
            f" | "
            f"{self.folder_name}"
            f" ({self.image_count} images)"
        )