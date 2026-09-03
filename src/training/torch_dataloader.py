"""DataLoader factory for batching :class:`LivestockDataset` samples."""

from __future__ import annotations

from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def create_dataloader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int = 1,
    shuffle: bool = False,
    num_workers: int = 0,
    drop_last: bool = False,
    generator: torch.Generator | None = None,
) -> DataLoader[dict[str, Any]]:
    """Create a standard PyTorch loader responsible only for batching samples."""
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=drop_last,
        generator=generator,
    )


__all__ = ["create_dataloader"]
