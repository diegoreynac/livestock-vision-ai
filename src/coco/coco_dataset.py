"""
COCO dataset models.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.dataset.enums import (
    DatasetType,
    Sex,
    ViewType,
)

from src.coco.models import (
    COCOAnnotation,
    COCOImage,
    SegmentationMask,
)


# ==========================================================
# Cow Sample
# ==========================================================

@dataclass(slots=True)
class CowSample:
    """
    Complete sample representing one cow image.
    """

    animal_id: int

    dataset: DatasetType

    sex: Sex

    view: ViewType

    weight: float

    image: COCOImage

    mask: SegmentationMask

    annotation: COCOAnnotation

# ==========================================================
# COCO Dataset
# ==========================================================

@dataclass(slots=True)
class COCODataset:
    """
    Collection of cow samples.
    """

    samples: list[CowSample] = field(
        default_factory=list
    )

    def add(
        self,
        sample: CowSample
    ) -> None:
        """
        Add a sample to the dataset.
        """

        self.samples.append(sample)

    def get_sample(
        self,
        animal_id: int
    ) -> CowSample | None:
        """
        Find a sample by animal ID.
        """

        for sample in self.samples:

            if sample.animal_id == animal_id:

                return sample

        return None

    def __len__(self) -> int:

        return len(
            self.samples
        )
    
    def __iter__(self):

        return iter(
            self.samples
        )
    
    def __getitem__(
        self,
        index: int
    ) -> CowSample:

        return self.samples[index]
    
    @property
    def images(self) -> list[COCOImage]:

        return [

            sample.image

            for sample in self.samples

        ]
    
    @property
    def annotations(self) -> list[COCOAnnotation]:

        return [

            sample.annotation

            for sample in self.samples

        ]
    
    @property
    def masks(self) -> list[SegmentationMask]:

        return [

            sample.mask

            for sample in self.samples

        ]
    
