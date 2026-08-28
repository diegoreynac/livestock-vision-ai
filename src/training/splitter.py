from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Dict
import random

from src.training.samples import TrainingSample


@dataclass(slots=True)
class DatasetSplit:
    train: List[TrainingSample]
    validation: List[TrainingSample]
    test: List[TrainingSample]


class DatasetSplitter:
    """Split a collection of TrainingSample objects at the animal (sample) level.

    Behavior:
    - Deterministic when a seed is provided (uses random.Random(seed)).
    - Preserves TrainingSample objects (side/rear pairing and metadata remain intact).
    - Validates inputs and raises ValueError for invalid ratios or duplicated animal_ids.
    - Returns empty lists for an empty input collection.
    """

    def __init__(self, train: float = 0.7, validation: float = 0.2, test: float = 0.1, seed: int = 42) -> None:
        self._validate_ratios(train, validation, test)
        self.train_ratio = float(train)
        self.validation_ratio = float(validation)
        self.test_ratio = float(test)
        self.seed = int(seed)

    @staticmethod
    def _validate_ratios(train: float, validation: float, test: float) -> None:
        for name, value in ("train", train), ("validation", validation), ("test", test):
            if value is None or not isinstance(value, (int, float)):
                raise ValueError(f"Split ratio '{name}' must be a number.")
            if value < 0.0 or value > 1.0:
                raise ValueError(f"Split ratio '{name}' must be between 0 and 1.")

        total = float(train) + float(validation) + float(test)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"Split ratios must sum to 1.0; got {total}.")

    def split(self, samples: Iterable[TrainingSample]) -> DatasetSplit:
        samples_list = list(samples)

        if not samples_list:
            return DatasetSplit(train=[], validation=[], test=[])

        # Validate animal_id presence and build index
        animal_map: Dict[str, TrainingSample] = {}
        duplicates: List[str] = []

        for s in samples_list:
            if s.animal_id is None:
                raise ValueError(f"TrainingSample with missing animal_id: {s}")
            key = str(s.animal_id).strip()
            if key == "":
                raise ValueError(f"TrainingSample with invalid/empty animal_id: {s}")
            if key in animal_map:
                duplicates.append(key)
            else:
                animal_map[key] = s

        if duplicates:
            # Report duplicates explicitly; do not silently resolve
            uniq = sorted(set(duplicates))
            raise ValueError(f"Duplicate animal_id values found for animals: {uniq}")

        # Use a deterministic ordering of animal IDs before shuffling so
        # the same seed produces the same splits regardless of input order.
        animal_ids = sorted(animal_map.keys())

        rng = random.Random(self.seed)
        rng.shuffle(animal_ids)

        n = len(animal_ids)
        n_train = int(round(self.train_ratio * n))
        n_validation = int(round(self.validation_ratio * n))
        # Ensure total sums to n by assigning remainder to test
        n_assigned = n_train + n_validation
        n_test = n - n_assigned

        # Edge-case adjustments: if rounding produced negative or overshoot, clamp
        if n_test < 0:
            # Reduce validation first, then train
            excess = -n_test
            reduce_val = min(excess, n_validation)
            n_validation -= reduce_val
            excess -= reduce_val
            if excess > 0:
                reduce_train = min(excess, n_train)
                n_train -= reduce_train
                excess -= reduce_train
            n_test = n - (n_train + n_validation)

        # Build splits by slicing the shuffled list
        train_ids = set(animal_ids[:n_train])
        validation_ids = set(animal_ids[n_train : n_train + n_validation])
        test_ids = set(animal_ids[n_train + n_validation :])

        # Final overlap check (should not happen)
        if train_ids & validation_ids or train_ids & test_ids or validation_ids & test_ids:
            raise RuntimeError("Overlap detected between splits after assignment.")

        train_samples = [animal_map[a] for a in animal_ids if a in train_ids]
        validation_samples = [animal_map[a] for a in animal_ids if a in validation_ids]
        test_samples = [animal_map[a] for a in animal_ids if a in test_ids]

        return DatasetSplit(train=train_samples, validation=validation_samples, test=test_samples)

    @staticmethod
    def validate_no_leakage(split: DatasetSplit) -> None:
        ids = []
        ids.extend([s.animal_id for s in split.train])
        ids.extend([s.animal_id for s in split.validation])
        ids.extend([s.animal_id for s in split.test])
        cleaned = [str(i).strip() for i in ids]
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("Animal ID appears in more than one split.")
