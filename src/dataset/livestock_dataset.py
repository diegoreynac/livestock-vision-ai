"""
Livestock dataset.

Represents the complete livestock image dataset.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.dataset.enums import DatasetType
from src.dataset.enums import Sex
from src.dataset.enums import View

from src.dataset.models import (
    ImageFolder,
    ImageRecord
)


class LivestockDataset:
    """
    Represents the complete livestock dataset.
    """

    def __init__(self) -> None:

        self._folders: list[ImageFolder] = []

        self._records: list[ImageRecord] = []

    # =====================================================
    # Properties
    # =====================================================

    @property
    def folders(self) -> list[ImageFolder]:

        return self._folders

    @property
    def records(self) -> list[ImageRecord]:

        return self._records
    
    @property
    def unique_records(self) -> list[ImageRecord]:
        """
        Returns one record per animal.

        If multiple images exist for the same animal,
        the first occurrence is returned.
        """

        unique = {}

        for record in self._records:

            if record.animal_id not in unique:

                unique[record.animal_id] = record

        return list(unique.values())
    
    @property
    def unique_weights(self) -> list[float]:
        """
        Returns one weight per animal.
        """

        return [

            record.weight_kg

            for record in self.unique_records

        ]
    
    @property
    def unique_animal_count(self) -> int:

        return len(self.unique_records)
    
    

    # =====================================================
    # Add Data
    # =====================================================

    def add_folder(
        self,
        folder: ImageFolder
    ) -> None:

        self._folders.append(folder)

    def add_record(
    self,
    folder: ImageFolder,
    record: ImageRecord
    ) -> None:
        """
        Adds one image to the dataset.

        This is the only method that should insert new
        ImageRecord objects into the dataset.
        """

        record.folder = folder

        folder.records.append(record)

        self._records.append(record)

    # =====================================================
    # Dataset Information
    # =====================================================

    @property
    def folder_count(self) -> int:

        return len(self._folders)

    @property
    def image_count(self) -> int:

        return len(self._records)

    @property
    def animal_count(self) -> int:

        return len(

            {

                record.animal_id

                for record in self._records

            }

        )

    @property
    def male_count(self) -> int:

        return sum(

            record.is_male

            for record in self._records

        )

    @property
    def female_count(self) -> int:

        return sum(

            record.is_female

            for record in self._records

        )

    @property
    def side_count(self) -> int:

        return sum(

            record.is_side

            for record in self._records

        )

    @property
    def rear_count(self) -> int:

        return sum(

            record.is_rear

            for record in self._records

        )

    @property
    def weights(self) -> list[float]:

        return [

            record.weight_kg

            for record in self._records

        ]
    
    @property
    def datasets(self) -> list[str]:
        """
        Returns the dataset label for every image.
        """

        return [
            record.dataset.value
            for record in self._records
        ]


    @property
    def views(self) -> list[str]:
        """
        Returns the camera view for every image.
        """

        return [
            record.view.value
            for record in self._records
        ]


    @property
    def sexes(self) -> list[str]:
        """
        Returns the sex label for every image.
        """

        return [
            record.sex.value
            for record in self._records
        ]


    @property
    def animal_ids(self) -> list[str]:
        """
        Returns every animal identifier.
        """

        return [
            record.animal_id
            for record in self._records
        ]

    # =====================================================
    # Filters
    # =====================================================

    def by_dataset(
        self,
        dataset: DatasetType
    ) -> list[ImageRecord]:

        return [

            record

            for record in self._records

            if record.dataset == dataset

        ]

    def by_view(
        self,
        view: View
    ) -> list[ImageRecord]:

        return [

            record

            for record in self._records

            if record.view == view

        ]

    def by_sex(
        self,
        sex: Sex
    ) -> list[ImageRecord]:

        return [

            record

            for record in self._records

            if record.sex == sex

        ]
    


    # =====================================================
    # Export
    # =====================================================

    def to_dataframe(self) -> pd.DataFrame:

        return pd.DataFrame(

            [

                record.to_dict()

                for record in self._records

            ]

        )

    def export_csv(
        self,
        output: Path
    ) -> None:

        self.to_dataframe().to_csv(

            output,

            index=False

        )

    # =====================================================
    # Python Protocols
    # =====================================================

    def __len__(self):

        return len(self._records)

    def __iter__(self):

        return iter(self._records)

    def __getitem__(self, index):

        return self._records[index]