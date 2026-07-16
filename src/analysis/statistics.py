"""
Dataset statistics.

Computes descriptive statistics for the livestock dataset.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import (
    asdict,
    dataclass,
    field,
)


import numpy as np

from src.dataset.enums import (
    DatasetType,
    Sex,
    View,
)

from src.dataset.livestock_dataset import LivestockDataset
from src.dataset.models import ImageRecord

# ==========================================================
# GROUP STATISTICS
# ==========================================================

@dataclass(slots=True)
class GroupStatistics:
    """
    Statistics of a subset of the dataset.
    """

    name: str

    image_count: int

    animal_count: int

    minimum_weight: float

    maximum_weight: float

    mean_weight: float

    median_weight: float

    standard_deviation: float

    variance: float

    q1: float

    q3: float


# ==========================================================
# WEIGHT STATISTICS
# ==========================================================

@dataclass(slots=True)
class WeightStatistics:

    minimum: float

    maximum: float

    mean: float

    median: float

    standard_deviation: float

    variance: float

    q1: float

    q3: float

    iqr: float

# ==========================================================
# RESULTS
# ==========================================================

@dataclass(slots=True)
class DatasetStatisticsResult:
    """
    Stores all computed statistics required by the analysis
    and visualization modules.
    """

    # General
    image_count: int
    animal_count: int
    folder_count: int

    # Distributions
    dataset_distribution: Counter
    view_distribution: Counter
    sex_distribution: Counter

    # Weight statistics
    weight: WeightStatistics

    # Group summaries
    datasets: list[GroupStatistics]
    views: list[GroupStatistics]
    sexes: list[GroupStatistics]

    # Raw data used by visualization
    weights: list[float] = field(default_factory=list)

    dataset_labels: list[str] = field(default_factory=list)

    view_labels: list[str] = field(default_factory=list)

    sex_labels: list[str] = field(default_factory=list)

    # =====================================================
    # Export
    # =====================================================

    def to_dict(self) -> dict:
        """
        Convert the statistics result into a serializable dictionary.
        """

        return {

            "image_count": self.image_count,

            "animal_count": self.animal_count,

            "folder_count": self.folder_count,

            "dataset_distribution": dict(
                self.dataset_distribution
            ),

            "view_distribution": dict(
                self.view_distribution
            ),

            "sex_distribution": dict(
                self.sex_distribution
            ),

            "weight": asdict(self.weight),

            "datasets": [
                asdict(group)
                for group in self.datasets
            ],

            "views": [
                asdict(group)
                for group in self.views
            ],

            "sexes": [
                asdict(group)
                for group in self.sexes
            ],

            "weights": self.weights,

            "dataset_labels": self.dataset_labels,

            "view_labels": self.view_labels,

            "sex_labels": self.sex_labels

        }

# ==========================================================
# DATASET STATISTICS
# ==========================================================

class DatasetStatistics:

    def __init__(
        self,
        dataset: LivestockDataset,
    ) -> None:

        self.dataset = dataset

        self.results: DatasetStatisticsResult | None = None

        # =====================================================
    # Public API
    # =====================================================

    def compute(self) -> DatasetStatisticsResult:
        """
        Compute all dataset statistics.
        """

        weights = np.asarray(
            self.dataset.weights,
            dtype=np.float64
        )

        self.results = DatasetStatisticsResult(

            image_count=self.dataset.image_count,

            animal_count=self.dataset.animal_count,

            folder_count=self.dataset.folder_count,

            dataset_distribution=self._dataset_distribution(),

            view_distribution=self._view_distribution(),

            sex_distribution=self._sex_distribution(),

            weight=self._compute_weight_statistics(weights),

            datasets=self._dataset_statistics(),

            views=self._view_statistics(),

            sexes=self._sex_statistics(),

            weights=self.dataset.weights,

            dataset_labels=self.dataset.datasets,

            view_labels=self.dataset.views,

            sex_labels=self.dataset.sexes

        )

        return self.results

    # =====================================================
    # Distribution
    # =====================================================

    def _dataset_distribution(self) -> Counter:

        counter = Counter()

        for record in self.dataset:

            counter[record.dataset.value] += 1

        return counter


    def _view_distribution(self) -> Counter:

        counter = Counter()

        for record in self.dataset:

            counter[record.view.value] += 1

        return counter


    def _sex_distribution(self) -> Counter:

        counter = Counter()

        for record in self.dataset:

            counter[record.sex.value] += 1

        return counter


    # =====================================================
    # Weight Statistics
    # =====================================================

    def _compute_weight_statistics(
        self,
        weights: np.ndarray
    ) -> WeightStatistics:

        return WeightStatistics(

            minimum=float(np.min(weights)),

            maximum=float(np.max(weights)),

            mean=float(np.mean(weights)),

            median=float(np.median(weights)),

            standard_deviation=float(np.std(weights)),

            variance=float(np.var(weights)),

            q1=float(np.percentile(weights, 25)),

            q3=float(np.percentile(weights, 75)),

            iqr=float(
                np.percentile(weights, 75)
                - np.percentile(weights, 25)
            )

        )


    # =====================================================
    # Group Statistics
    # =====================================================

    def _group_statistics(
        self,
        name: str,
        records: list[ImageRecord]
    ) -> GroupStatistics:

        weights = np.asarray(
            [record.weight_kg for record in records],
            dtype=np.float64
        )

        return GroupStatistics(
            name=name,
            image_count=len({record.animal_id for record in records}),
            animal_count=len({record.animal_id for record in records}),
            minimum_weight=float(np.min(weights)),
            maximum_weight=float(np.max(weights)),
            mean_weight=float(np.mean(weights)),
            median_weight=float(np.median(weights)),
            standard_deviation=float(np.std(weights)),
            variance=float(np.var(weights)),
            q1=float(np.percentile(weights, 25)),
            q3=float(np.percentile(weights, 75)),
        )

    def _dataset_statistics(
        self
    ) -> list[GroupStatistics]:

        groups = []

        for dataset in DatasetType:

            records = self.dataset.by_dataset(dataset)

            if records:

                groups.append(

                    self._group_statistics(

                        dataset.value,

                        records

                    )

                )

        return groups


    def _view_statistics(
        self
    ) -> list[GroupStatistics]:

        groups = []

        for view in View:

            records = self.dataset.by_view(view)

            if records:

                groups.append(

                    self._group_statistics(

                        view.value,

                        records

                    )

                )

        return groups


    def _sex_statistics(
        self
    ) -> list[GroupStatistics]:

        groups = []

        for sex in Sex:

            records = self.dataset.by_sex(sex)

            if records:

                groups.append(

                    self._group_statistics(

                        sex.value,

                        records

                    )

                )

        return groups