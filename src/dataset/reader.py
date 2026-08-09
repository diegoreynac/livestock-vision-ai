"""
Dataset reader.

Loads the livestock dataset from disk.
"""

from __future__ import annotations

from pathlib import Path

from src.core.context import ProjectContext

from src.dataset.enums import (
    DatasetType,
    ImageExtension,
    View,
)

from src.dataset.models import (
    ImageFolder,
)

from src.dataset.livestock_dataset import (
    LivestockDataset,
)

from src.dataset.parser import (
    FilenameParser,
)


class DatasetReader:
    """
    Reads the livestock dataset from disk.
    """

    def __init__(
        self,
        context: ProjectContext,
    ) -> None:

        self.context = context

        self.parser = FilenameParser()

    # =====================================================
    # Public
    # =====================================================

    def load(self) -> LivestockDataset:

        self.context.logger.section("Reading Dataset")

        dataset = LivestockDataset()

        folders = self._discover_image_folders()

        for folder in folders:

            self._read_folder(dataset, folder)

            # Only keep folders that contain images
            if folder.image_count > 0:

                dataset.add_folder(folder)

        self.context.logger.section("Dataset Loaded")

        self.context.logger.key_value(
            "Folders",
            dataset.folder_count
        )

        self.context.logger.key_value(
            "Images",
            dataset.image_count
        )

        self.context.logger.key_value(
            "Animals",
            dataset.animal_count
        )

        return dataset

    # =====================================================
    # Private
    # =====================================================

    def _discover_image_folders(
        self,
    ) -> list[ImageFolder]:

        folders = []

        pixel_root = (
            self.context.config.pixel_dataset
        )

        for dataset_dir in sorted(pixel_root.iterdir()):

            if not dataset_dir.is_dir():
                continue

            dataset_type = DatasetType.from_string(
                dataset_dir.name
            )

            for folder in sorted(dataset_dir.iterdir()):

                images_path = folder / "images"

                if not images_path.exists():
                    continue

                view = View.from_string(
                    folder.name
                )

                coco_files = sorted(folder.glob("*.json"))

                annotation_file = (
                    coco_files[0]
                    if coco_files
                    else None
                )

                folders.append(

                    ImageFolder(

                        dataset=dataset_type,

                        folder_name=folder.name,

                        view=view,

                        path=images_path,

                        annotation_file=annotation_file,

                    )

                )

        return folders

    # -----------------------------------------------------

    def _read_folder(
        self,
        dataset: LivestockDataset,
        folder: ImageFolder,
    ) -> None:

        image_count = 0

        self.context.logger.info(
            f"{folder.dataset.value} | {folder.folder_name}"
        )

        for image_path in sorted(folder.path.iterdir()):

            if not image_path.is_file():
                continue

            if not ImageExtension.is_supported(
                image_path
            ):
                continue

            try:

                record = self.parser.parse(
                    folder,
                    image_path
                )

                dataset.add_record(
                    folder,
                    record
                )

                image_count += 1

            except Exception as ex:

                self.context.logger.warning(
                    f"Ignored image: {ex}"
                )

                # self.context.logger.exception(ex)

        self.context.logger.key_value(
            "Images",
            image_count
        )

        self.context.logger.info("")