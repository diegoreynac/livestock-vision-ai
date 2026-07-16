"""
Dataset utilities.
"""

from pathlib import Path
from typing import List

from src.dataset.models import DatasetFolder
from src.dataset.models import ImageRecord

from src.dataset.parser import parse_filename


# ==========================================================
# DISCOVER IMAGE FOLDERS
# ==========================================================

def discover_image_folders(dataset_root: Path) -> List[DatasetFolder]:

    folders = []

    pixel_root = dataset_root / "Pixel"

    if not pixel_root.exists():
        raise FileNotFoundError(pixel_root)

    for dataset_dir in sorted(pixel_root.iterdir()):

        if not dataset_dir.is_dir():
            continue

        for folder in sorted(dataset_dir.iterdir()):

            if not folder.is_dir():
                continue

            images_folder = folder / "images"

            if not images_folder.exists():
                continue

            folders.append(

                DatasetFolder(

                    dataset=dataset_dir.name,

                    folder=folder.name,

                    path=images_folder

                )

            )

    return folders


# ==========================================================
# READ DATASET
# ==========================================================

def read_dataset(dataset_root: Path) -> List[ImageRecord]:

    folders = discover_image_folders(dataset_root)

    records = []

    for folder in folders:

        for image in sorted(folder.path.iterdir()):

            if image.suffix.lower() not in (

                ".jpg",

                ".jpeg",

                ".png"

            ):

                continue

            try:

                record = parse_filename(

                    dataset=folder.dataset,

                    folder=folder.folder,

                    filepath=image

                )

                records.append(record)

            except Exception:

                continue

    return records