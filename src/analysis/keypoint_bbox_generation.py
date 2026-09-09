"""Generate alternative COCO annotations using keypoint-derived bounding boxes.

This module never modifies the source COCO annotation data. It creates a
separate COCO JSON structure where invalid/missing bounding boxes can be
replaced by candidate bounding boxes derived from valid keypoints.

The generated annotations are experimental alternatives and must not be
considered ground truth until explicitly approved for training.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

from src.analysis.bbox_annotation_audit import BBoxAnnotationAudit


class KeypointBBoxGenerator:
    """Generate a new COCO annotation set using keypoint-derived BBoxes."""

    def generate(
        self,
        data: dict[str, Any],
    ) -> tuple[dict[str, Any], int]:
        """Generate a new COCO dictionary without modifying ``data``.

        Only annotations with an invalid/missing BBox and a valid candidate
        BBox derived from keypoints are changed.

        Returns:
            A new COCO dictionary and the number of BBoxes replaced.
        """
        if not isinstance(data, dict):
            raise TypeError("COCO data must be a dictionary.")

        if "images" not in data or not isinstance(data["images"], list):
            raise ValueError("COCO data must contain an 'images' list.")

        if "annotations" not in data or not isinstance(data["annotations"], list):
            raise ValueError("COCO data must contain an 'annotations' list.")

        # Deep copy guarantees that neither the top-level structure nor
        # nested annotation dictionaries are modified.
        generated = copy.deepcopy(data)

        audit = BBoxAnnotationAudit()
        report = audit.audit(data)

        candidates_by_annotation_id: dict[int, tuple[float, float, float, float]] = {}

        for image in report.images:
            for annotation in image.annotations:
                # We only replace invalid/missing BBoxes.
                if annotation.bbox_is_valid:
                    continue

                candidate = annotation.candidate_bbox
                if candidate is None:
                    continue

                if annotation.annotation_id is None:
                    continue

                candidates_by_annotation_id[annotation.annotation_id] = (
                    candidate.x_min,
                    candidate.y_min,
                    candidate.width,
                    candidate.height,
                )

        replaced_count = 0

        for annotation in generated["annotations"]:
            annotation_id = annotation.get("id")

            if annotation_id not in candidates_by_annotation_id:
                continue

            bbox = candidates_by_annotation_id[annotation_id]

            annotation["bbox"] = list(bbox)

            # These fields describe the generated annotation without
            # changing the original source file.
            annotation["bbox_source"] = "keypoints"
            annotation["bbox_method"] = "min_max_visible_keypoints"

            # COCO area corresponds to width * height.
            annotation["area"] = bbox[2] * bbox[3]

            # Mark the generated BBox as present in the alternative dataset.
            annotation["isbbox"] = True

            replaced_count += 1

        return generated, replaced_count

    def generate_file(
        self,
        source_file: Path,
        destination_file: Path,
    ) -> int:
        """Generate an alternative COCO JSON file.

        The source file is read-only. The destination is created separately.

        Returns:
            Number of BBoxes replaced.
        """
        source_file = Path(source_file)
        destination_file = Path(destination_file)

        with source_file.open("r", encoding="utf-8") as file:
            data = json.load(file)

        generated, replaced_count = self.generate(data)

        destination_file.parent.mkdir(parents=True, exist_ok=True)

        with destination_file.open("w", encoding="utf-8") as file:
            json.dump(
                generated,
                file,
                indent=2,
                ensure_ascii=False,
            )
            file.write("\n")

        return replaced_count


__all__ = ["KeypointBBoxGenerator"]