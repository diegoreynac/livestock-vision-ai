from __future__ import annotations

import unittest
from typing import List

from src.training.samples import TrainingSample
from src.dataset.enums import Sex
from src.training.splitter import DatasetSplitter
from pathlib import Path


def make_sample(animal_id: str, side: bool = True, rear: bool = True, sex: Sex = Sex.FEMALE, weight: float = 100.0) -> TrainingSample:
    side_path = Path(f"/tmp/{animal_id}_side.jpg") if side else None
    rear_path = Path(f"/tmp/{animal_id}_rear.jpg") if rear else None
    return TrainingSample(
        animal_id=animal_id,
        side_image=side_path,
        rear_image=rear_path,
        side_annotation=None,
        rear_annotation=None,
        sex=sex,
        weight_kg=weight,
    )


class SplitterTests(unittest.TestCase):

    def test_empty_dataset_returns_empty_splits(self):
        splitter = DatasetSplitter()
        split = splitter.split([])
        self.assertEqual(len(split.train), 0)
        self.assertEqual(len(split.validation), 0)
        self.assertEqual(len(split.test), 0)

    def test_default_ratios_approximate_counts(self):
        # Create 20 animals
        samples = [make_sample(f"A{i}") for i in range(20)]
        splitter = DatasetSplitter()
        split = splitter.split(samples)
        total = len(split.train) + len(split.validation) + len(split.test)
        self.assertEqual(total, 20)
        # Expect 70/20/10 -> 14/4/2 after rounding
        self.assertEqual(len(split.train), 14)
        self.assertEqual(len(split.validation), 4)
        self.assertEqual(len(split.test), 2)

    def test_deterministic_with_same_seed(self):
        samples = [make_sample(f"B{i}") for i in range(30)]
        s1 = DatasetSplitter(seed=123)
        s2 = DatasetSplitter(seed=123)
        split1 = s1.split(samples)
        split2 = s2.split(samples)
        ids1 = ([s.animal_id for s in split1.train], [s.animal_id for s in split1.validation], [s.animal_id for s in split1.test])
        ids2 = ([s.animal_id for s in split2.train], [s.animal_id for s in split2.validation], [s.animal_id for s in split2.test])
        self.assertEqual(ids1, ids2)

    def test_different_seeds_can_differ(self):
        samples = [make_sample(f"C{i}") for i in range(20)]
        s1 = DatasetSplitter(seed=1)
        s2 = DatasetSplitter(seed=2)
        split1 = s1.split(samples)
        split2 = s2.split(samples)
        # It's possible (rare) they match; check that at least one split differs
        same = (
            [s.animal_id for s in split1.train] == [s.animal_id for s in split2.train]
            and [s.animal_id for s in split1.validation] == [s.animal_id for s in split2.validation]
            and [s.animal_id for s in split1.test] == [s.animal_id for s in split2.test]
        )
        self.assertFalse(same)

    def test_no_animal_in_more_than_one_split(self):
        samples = [make_sample(f"D{i}") for i in range(25)]
        splitter = DatasetSplitter()
        split = splitter.split(samples)
        all_ids = [s.animal_id for s in split.train] + [s.animal_id for s in split.validation] + [s.animal_id for s in split.test]
        self.assertEqual(len(all_ids), len(set(all_ids)))

    def test_side_rear_pairing_preserved(self):
        samples = [make_sample(f"E{i}", side=True, rear=True) for i in range(10)]
        splitter = DatasetSplitter()
        split = splitter.split(samples)
        for s in split.train + split.validation + split.test:
            # Both views should remain with the sample
            self.assertTrue((s.side_image is not None) and (s.rear_image is not None))

    def test_metadata_preserved(self):
        s = make_sample("X1", sex=Sex.MALE, weight=250.5)
        splitter = DatasetSplitter()
        split = splitter.split([s])
        # Single sample should go to train by rounding rules
        self.assertEqual(len(split.train), 1)
        out = split.train[0]
        self.assertEqual(out.sex, Sex.MALE)
        self.assertEqual(out.weight_kg, 250.5)

    def test_invalid_ratios_raise(self):
        with self.assertRaises(ValueError):
            DatasetSplitter(train=0.5, validation=0.6, test=-0.1)
        with self.assertRaises(ValueError):
            DatasetSplitter(train=0.5, validation=0.6, test=0.0)

    def test_single_view_animals_handled(self):
        samples = [make_sample(f"S{i}", side=(i % 2 == 0), rear=(i % 2 == 1)) for i in range(7)]
        # Builder might have filtered these out; splitter should accept them and preserve
        splitter = DatasetSplitter()
        split = splitter.split(samples)
        total = len(split.train) + len(split.validation) + len(split.test)
        self.assertEqual(total, 7)


    def test_small_dataset_edge_cases(self):
        # Verify for n=1,2,3 that no samples are dropped and no leaks occur
        for n in (1, 2, 3):
            samples = [make_sample(f"small{n}_{i}") for i in range(n)]
            splitter = DatasetSplitter(seed=7)
            split = splitter.split(samples)
            total = len(split.train) + len(split.validation) + len(split.test)
            self.assertEqual(total, n, f"Total samples for n={n} should be {n}")
            # Ensure no duplicate animal across splits
            all_ids = [s.animal_id for s in split.train] + [s.animal_id for s in split.validation] + [s.animal_id for s in split.test]
            self.assertEqual(len(all_ids), len(set(all_ids)), f"No duplicate animal_ids for n={n}")

    def test_validate_no_leakage_raises_on_overlap(self):
        # Construct a DatasetSplit that intentionally leaks the same animal into two splits
        from src.training.splitter import DatasetSplit
        sample = make_sample("leak1")
        ds = DatasetSplit(train=[sample], validation=[], test=[sample])
        with self.assertRaises(ValueError):
            DatasetSplitter.validate_no_leakage(ds)

    def test_deterministic_independent_of_input_order(self):
        # Build the same logical samples in two different input orders and assert splits match for same seed
        samples = [make_sample(f"ord{i}") for i in range(20)]
        ordered = list(samples)
        reversed_order = list(reversed(samples))
        s1 = DatasetSplitter(seed=42)
        s2 = DatasetSplitter(seed=42)
        split1 = s1.split(ordered)
        split2 = s2.split(reversed_order)
        ids1 = ([s.animal_id for s in split1.train], [s.animal_id for s in split1.validation], [s.animal_id for s in split1.test])
        ids2 = ([s.animal_id for s in split2.train], [s.animal_id for s in split2.validation], [s.animal_id for s in split2.test])
        self.assertEqual(ids1, ids2)


if __name__ == "__main__":
    unittest.main()
