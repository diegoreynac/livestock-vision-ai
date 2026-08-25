"""
COCO dataset reader.

Enriches an existing LivestockDataset using
COCO annotations and segmentation masks.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Any

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


RecordKey = tuple[str, str, str]


class COCOReader:
    """
    Reads COCO annotations and enriches an existing
    LivestockDataset.
    """

    # Cardinalities observed by COCOAudit in the current dataset. They are
    # diagnostic expectations, not an acceptance constraint: a COCO keypoint
    # payload is structurally valid whenever it follows the flat 3N form.
    _KNOWN_KEYPOINT_CARDINALITIES = frozenset({4, 6, 9, 23})

    def __init__(
        self,
        context: ProjectContext,
    ) -> None:

        self.context = context

        self.config = context.config

        self.logger = context.logger

        self.data: dict = {}

        self.record_index: dict[RecordKey, ImageRecord] = {}

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

            if record.folder is None:
                self._error(
                    f"Record without folder cannot be indexed: {record.filename}"
                )
                continue

            key = self._record_key(
                record.folder,
                record.filename,
            )

            if key in self.record_index:
                self._warning(
                    "Duplicate record filename in the same context: "
                    f"{self._format_record_key(key)}"
                )
                continue

            self.record_index[key] = record

        self.statistics.images_indexed = len(
            self.record_index
        )

        self.logger.info(
            f"Indexed {len(self.record_index)} images."
        ) 

    @staticmethod
    def _record_key(
        folder: ImageFolder,
        filename: str,
    ) -> RecordKey:
        """Build the contextual identity used for a physical image record."""

        return (
            folder.dataset.value,
            folder.folder_name,
            filename,
        )

    @staticmethod
    def _format_record_key(key: RecordKey) -> str:
        """Format a contextual record key for diagnostic logging."""

        dataset, folder, filename = key
        return f"{dataset}/{folder}/{filename}"

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

        image_index: dict[int, dict] = {}
        filenames_seen: set[str] = set()

        for image in data["images"]:

            image_id = image.get("id")
            filename = image.get("file_name")

            if image_id in image_index:
                self._warning(
                    f"{folder.folder_name}: Duplicate image id {image_id}."
                )
                continue

            if not isinstance(filename, str):
                self._warning(
                    f"{folder.folder_name}: Image id {image_id} has no valid filename."
                )
                continue

            if filename in filenames_seen:
                self._warning(
                    f"{folder.folder_name}: Duplicate JSON filename '{filename}'."
                )

            filenames_seen.add(filename)
            image_index[image_id] = image

        # --------------------------------------------------
        # Process every annotation
        # --------------------------------------------------

        annotated_image_ids: set[int] = set()

        for item in data["annotations"]:

            image_id = item.get("image_id")
            image = image_index.get(image_id)

            if image is None:

                self._warning(
                    f"{folder.folder_name}: Image id {image_id} not found "
                    f"for annotation {item.get('id')}."
                )

                continue

            if image_id in annotated_image_ids:
                self._warning(
                    f"{folder.folder_name}: Multiple annotations for image id "
                    f"{image_id}; annotation {item.get('id')} was skipped."
                )
                continue

            annotated_image_ids.add(image_id)

            filename = image["file_name"]

            record_key = self._record_key(folder, filename)
            record = self.record_index.get(record_key)

            if record is None:

                self._warning(
                    "Physical image not found for JSON annotation: "
                    f"{self._format_record_key(record_key)}"
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
                    "Existing annotation was preserved; incoming annotation "
                    f"{item.get('id')} was skipped for "
                    f"{self._format_record_key(record_key)}."
                )

                continue

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

        if not self._is_valid_keypoint_payload(keypoint_values):

            self._warning(
                f"Invalid keypoints in annotation {item.get('id')}: "
                "expected a flat [x, y, visibility] * N payload."
            )

            return None

        keypoint_count = len(keypoint_values) // 3

        self._log_unusual_keypoint_cardinality(
            item.get("id"),
            keypoint_count,
        )

        keypoints = self._create_keypoints(
            keypoint_values
        )

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

    def _is_valid_keypoint_payload(
        self,
        values: Any,
    ) -> bool:
        """Validate COCO's flat ``[x, y, visibility] * N`` representation."""

        if not isinstance(values, list) or len(values) % 3 != 0:
            return False

        for index in range(0, len(values), 3):

            x, y, visibility = values[index:index + 3]

            if not isinstance(x, (int, float)):
                return False

            if not isinstance(y, (int, float)):
                return False

            try:
                KeypointVisibility(visibility)
            except (TypeError, ValueError):
                return False

        return True

    def _log_unusual_keypoint_cardinality(
        self,
        annotation_id: int | None,
        keypoint_count: int,
    ) -> None:
        """Report, without rejecting, a valid but unseen cardinality."""

        if keypoint_count in self._KNOWN_KEYPOINT_CARDINALITIES:
            return

        self._warning(
            f"Annotation {annotation_id}: unusual but valid keypoint "
            f"cardinality ({keypoint_count})."
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

