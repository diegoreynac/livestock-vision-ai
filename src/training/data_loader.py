"""Training data loading entry points."""

from __future__ import annotations

from src.coco.reader import COCOReader
from src.core.context import ProjectContext
from src.dataset.reader import DatasetReader
from src.training.builder import DatasetBuilder
from src.training.samples import TrainingSample


def load_training_samples(
    context: ProjectContext,
) -> list[TrainingSample]:
    """Load the configured dataset and build training samples.

    The pipeline is intentionally kept as a thin orchestration layer:

    1. Read physical images and metadata into ImageRecord objects.
    2. Enrich ImageRecord objects with COCO annotations.
    3. Group records by animal and require both Side and Rear views.
    """
    dataset_reader = DatasetReader(context)

    dataset = dataset_reader.load()

    coco_reader = COCOReader(context)

    coco_reader.enrich(dataset)

    builder = DatasetBuilder(
        require_both_views=True,
    )

    return builder.build(dataset.records)