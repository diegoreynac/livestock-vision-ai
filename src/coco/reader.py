"""
COCO dataset reader.

Enriches an existing LivestockDataset using
COCO annotations and segmentation masks.
"""

from __future__ import annotations

import json

from pathlib import Path

from src.core.context import ProjectContext
from src.coco.statistics import COCOStatistics

from src.dataset.livestock_dataset import LivestockDataset
from src.dataset.models import (
    ImageFolder,
    ImageRecord,
)

from src.coco.models import (
    COCOAnnotation,
    COCOBoundingBox,
    COCOCategory,
    COCOImage,
    COCOKeypoint
)

from src.coco.enums import KeypointVisibility


class COCOReader:
    """
    Reads COCO annotations and enriches an existing
    LivestockDataset.
    """

    def __init__(
        self,
        context: ProjectContext,
    ) -> None:

        self.context = context

        self.config = context.config

        self.logger = context.logger

        self.data: dict = {}

        self.record_index: dict[
            str,
            ImageRecord
        ] = {}

        self.statistics = COCOStatistics()

    # --------------------------------------------------
    # Public API
    # --------------------------------------------------

    def enrich(
        self,
        dataset: LivestockDataset,
    ) -> None:
        """
        Enrich the dataset with COCO annotations.
        """

        self.statistics.reset()


        self._build_record_index(dataset)

        for folder in dataset.folders:

            if not folder.has_coco:

                self.logger.warning(
                    f"COCO annotations not found for {folder.folder_name}"
                )

                continue

            self._process_folder(folder)

        self._validate_dataset(dataset)

        self.logger.section(
            "COCO Summary"
        )

        self.logger.info(
            str(self.statistics)
        )

        self.logger.info(
            "COCO dataset successfully loaded."
        )

    def _validate_dataset(
        self,
        dataset: LivestockDataset,
    ) -> None:
        """
        Validate loaded annotations.
        """

        total = 0
        annotated = 0

        missing = total - annotated

        self.statistics.missing_annotations = missing

        for record in dataset:

            total += 1

            if record.annotation is not None:

                annotated += 1

        missing = total - annotated

        self.logger.section(
            "COCO Summary"
        )

        self.logger.info(
            f"Images            : {total}"
        )

        self.logger.info(
            f"Annotations loaded: {annotated}"
        )

        self.logger.info(
            f"Missing annotations: {missing}"
        )

        if missing > 0:

            self.logger.warning(
                f"{missing} images have no annotation."
            )

    def _build_record_index(
        self,
        dataset: LivestockDataset,
    ) -> None:
        """
        Build a lookup table using image filenames.
        """

        self.record_index.clear()

        for record in dataset:

            self.record_index[
                record.filename
            ] = record

            self.logger.info(
                f"Records indexed: {len(self.record_index)}"
            )

        self.statistics.images_indexed = len(
            self.record_index
        )

        self.logger.info(
            f"Indexed {len(self.record_index)} images."
        ) 

    def _process_folder(
        self,
        folder: ImageFolder,
    ) -> None:
        """
        Process one COCO folder.
        """

        self.statistics.folders_processed += 1

        self.logger.info(
            f"Processing {folder.folder_name}"
        )

        data = self._load_json(folder.annotation_file)

        if not data:
            return

        if not self._validate_json(data):
            return

        self.logger.info(
            f"Images in JSON: {len(data['images'])}"
        )

        categories = self._read_categories(data)

        self._attach_annotations(
            folder,
            data,
            categories,
        )

    def _read_categories(
        self,
        data: dict,
    ) -> dict[int, COCOCategory]:
        """
        Read COCO categories.
        """

        categories = {}

        for item in data["categories"]:

            category = COCOCategory(
                id=item["id"],
                name=item["name"],
                supercategory=item.get("supercategory"),
            )

            categories[category.id] = category

        return categories
    
    def _load_json(
        self,
        annotation_file: Path,
    ) -> dict:
        """
        Load one COCO annotation file.
        """

        if not annotation_file.exists():

            self.logger.error(
                f"Annotation file not found: {annotation_file}"
            )

            return {}

        try:

            with annotation_file.open(
                "r",
                encoding="utf-8",
            ) as file:

                return json.load(file)

        except json.JSONDecodeError as error:

            self.logger.error(
                f"Invalid JSON file: {annotation_file}"
            )

            self.logger.error(str(error))

        except Exception as error:

            self.logger.error(str(error))

        return {}

    def _attach_annotations(
        self,
        folder: ImageFolder,
        data: dict,
        categories: dict[int, COCOCategory],
    ) -> None:
        """
        Attach COCO annotations to ImageRecord objects.
        """

        # --------------------------------------------------
        # Build image lookup table
        # --------------------------------------------------

        image_index = {}

        for image in data["images"]:

            image_index[image["id"]] = image

        # --------------------------------------------------
        # Process every annotation
        # --------------------------------------------------

        for item in data["annotations"]:

            image = image_index.get(item.get("image_id"))

            if image is None:

                self.logger.warning(
                    f"Image id {item.get('image_id')} not found."
                )

                continue

            filename = image["file_name"]

            record = self.record_index.get(filename)

            if filename not in self.record_index:

                self.logger.error(
                    f"Filename '{filename}' not found."
                )

                for key in list(self.record_index.keys())[:20]:
                    self.logger.error(f"Key -> '{key}'")

            # self.logger.info(
            #     f"JSON filename: {filename}"
            # )

            # self.logger.info(
            #     f"Found: {record is not None}"
            # )

            if filename not in self.record_index:

                self.logger.error(
                    f"Filename '{filename}' not found."
                )

                for key in list(self.record_index.keys())[:20]:
                    self.logger.error(f"Key -> '{key}'")

           
            if record is None:

                self.logger.warning(
                    f"Image not found: {filename}"
                )

                continue

            if item["category_id"] not in categories:

                self._warning(
                    f"Unknown category {item['category_id']}"
                )

                continue

            annotation = self._create_annotation(item)

            if annotation is None:
                self.logger.error(
                    f"Annotation could not be created for {filename}"
                )
                continue

            if record.annotation is not None:

                self._warning(
                    f"{record.filename}: Duplicate annotation."
                )

            record.annotation = annotation

            if record.annotation is None:
                self.logger.error(
                    f"Assignment failed for {record.filename}"
                )

            self.statistics.annotations_loaded += 1

    def _create_annotation(
        self,
        item: dict,
    ) -> COCOAnnotation | None:
        """
        Create a COCOAnnotation object from a JSON annotation.
        """

        bbox_values = item.get("bbox")

        if bbox_values is None or len(bbox_values) != 4:

            self._warning(...)

            return None

        bbox = self._create_bbox(bbox_values)

        if bbox is None:

            self._warning(
                f"Invalid bbox in annotation {item.get('id')}"
            )

            return None

        keypoint_values = item.get("keypoints", [])

        if len(keypoint_values) % 3 != 0:

            self._warning(...)

            return None

        keypoints = self._create_keypoints(
            keypoint_values
        )

        if len(keypoints) % 3 != 0:

            self._warning(
                f"Invalid keypoints in annotation {item.get('id')}"
            )

            return None

        return COCOAnnotation(

            id=item["id"],

            image_id=item["image_id"],

            category_id=item["category_id"],

            bbox=bbox,

            segmentation=item.get(
                "segmentation",
                [],
            ),

            keypoints=keypoints,

            area=item.get(
                "area",
                0.0,
            ),

            iscrowd=bool(
                item.get(
                    "iscrowd",
                    0,
                )
            ),
        )

    def _create_bbox(
        self,
        bbox: list[float],
    ) -> COCOBoundingBox:
        """
        Create a bounding box object.
        """

        return COCOBoundingBox(

            x=bbox[0],

            y=bbox[1],

            width=bbox[2],

            height=bbox[3],
        )
    
    def _create_keypoints(
        self,
        values: list[float],
    ) -> list[COCOKeypoint]:
        """
        Create COCO keypoints.
        """

        keypoints = []

        for index in range(
            0,
            len(values),
            3,
        ):

            keypoint = COCOKeypoint(

                keypoint_id=index // 3,

                x=values[index],

                y=values[index + 1],

                visibility=KeypointVisibility(
                    values[index + 2]
                ),
            )

            keypoints.append(
                keypoint
            )

        return keypoints
    
    def _validate_json(
        self,
        data: dict,
    ) -> bool:
        """
        Validate COCO JSON structure.
        """

        _REQUIRED_JSON_KEYS = (
            "images",
            "annotations",
            "categories",
        )

        for key in _REQUIRED_JSON_KEYS:

            if key not in data:

                self.logger.error(
                    f"Missing key '{key}' in COCO file."
                )

                return False

        return True
    
    def _warning(
        self,
        message: str,
    ) -> None:

        self.statistics.warnings += 1

        self.logger.warning(message)

    def _error(
        self,
        message: str,
    ) -> None:

        self.statistics.errors += 1

        self.logger.error(message)

    