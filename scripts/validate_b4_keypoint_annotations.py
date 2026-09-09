"""
Validate B4 keypoint-derived COCO bounding boxes.

This script is read-only:
- It never modifies the source JSON files.
- It validates bbox structure and image boundaries.
- It verifies that each generated bbox matches the visible keypoints.
- It reports anomalous annotations explicitly.

Input:
    output/keypoint_bbox_annotations/coco_b4_side_keypoint_bbox.json
    output/keypoint_bbox_annotations/coco_b4_rear_keypoint_bbox.json
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILES = {
    "B4 Side": (
        PROJECT_ROOT
        / "output"
        / "keypoint_bbox_annotations"
        / "coco_b4_side_keypoint_bbox.json"
    ),
    "B4 Rear": (
        PROJECT_ROOT
        / "output"
        / "keypoint_bbox_annotations"
        / "coco_b4_rear_keypoint_bbox.json"
    ),
}


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def get_visible_keypoints(
    keypoints: list[Any],
) -> list[tuple[float, float]]:
    """
    Return keypoints whose visibility flag is greater than zero.
    """

    visible = []

    if len(keypoints) % 3 != 0:
        return visible

    for index in range(0, len(keypoints), 3):
        x = keypoints[index]
        y = keypoints[index + 1]
        visibility = keypoints[index + 2]

        try:
            x = float(x)
            y = float(y)
            visibility = float(visibility)
        except (TypeError, ValueError):
            continue

        if (
            visibility > 0
            and math.isfinite(x)
            and math.isfinite(y)
        ):
            visible.append((x, y))

    return visible


def expected_bbox(
    keypoints: list[Any],
) -> tuple[float, float, float, float] | None:
    """
    Calculate the expected COCO bbox from visible keypoints.

    Returns:
        (x, y, width, height)
    """

    visible = get_visible_keypoints(keypoints)

    if len(visible) < 2:
        return None

    xs = [point[0] for point in visible]
    ys = [point[1] for point in visible]

    x_min = min(xs)
    y_min = min(ys)
    x_max = max(xs)
    y_max = max(ys)

    width = x_max - x_min
    height = y_max - y_min

    if width <= 0 or height <= 0:
        return None

    return x_min, y_min, width, height


def approximately_equal(
    first: float,
    second: float,
    tolerance: float = 1e-6,
) -> bool:
    return math.isclose(
        first,
        second,
        rel_tol=0.0,
        abs_tol=tolerance,
    )


def validate_dataset(
    dataset_name: str,
    path: Path,
) -> dict[str, Any]:

    data = load_json(path)

    images = {
        image["id"]: image
        for image in data.get("images", [])
    }

    annotations = data.get("annotations", [])

    results = {
        "dataset": dataset_name,
        "images": len(images),
        "annotations": len(annotations),
        "invalid_structure": [],
        "out_of_bounds": [],
        "non_positive": [],
        "bbox_mismatch": [],
        "missing_metadata": [],
        "valid": 0,
    }

    for annotation in annotations:

        annotation_id = annotation.get("id")
        image_id = annotation.get("image_id")

        image = images.get(image_id)

        if image is None:
            results["invalid_structure"].append({
                "annotation_id": annotation_id,
                "reason": "image_not_found",
                "image_id": image_id,
            })
            continue

        width = float(image["width"])
        height = float(image["height"])

        bbox = annotation.get("bbox")

        # --------------------------------------------------
        # Basic bbox structure
        # --------------------------------------------------

        if (
            not isinstance(bbox, list)
            or len(bbox) != 4
        ):
            results["invalid_structure"].append({
                "annotation_id": annotation_id,
                "reason": "bbox_must_have_four_values",
                "bbox": bbox,
            })
            continue

        try:
            x, y, bbox_width, bbox_height = map(float, bbox)
        except (TypeError, ValueError):
            results["invalid_structure"].append({
                "annotation_id": annotation_id,
                "reason": "bbox_contains_non_numeric_values",
                "bbox": bbox,
            })
            continue

        values = [x, y, bbox_width, bbox_height]

        if not all(math.isfinite(value) for value in values):
            results["invalid_structure"].append({
                "annotation_id": annotation_id,
                "reason": "bbox_contains_non_finite_values",
                "bbox": bbox,
            })
            continue

        # --------------------------------------------------
        # Positive dimensions
        # --------------------------------------------------

        if bbox_width <= 0 or bbox_height <= 0:
            results["non_positive"].append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "bbox": bbox,
            })
            continue

        # --------------------------------------------------
        # Image boundary validation
        # --------------------------------------------------

        x2 = x + bbox_width
        y2 = y + bbox_height

        if (
            x < 0
            or y < 0
            or x2 > width
            or y2 > height
        ):
            results["out_of_bounds"].append({
                "annotation_id": annotation_id,
                "image_id": image_id,
                "file_name": image.get("file_name"),
                "image_width": width,
                "image_height": height,
                "bbox": bbox,
                "bbox_xyxy": [x, y, x2, y2],
            })

        # --------------------------------------------------
        # Generated metadata
        # --------------------------------------------------

        if (
            annotation.get("bbox_source") != "keypoints"
            or annotation.get("bbox_method")
            != "min_max_visible_keypoints"
        ):
            results["missing_metadata"].append({
                "annotation_id": annotation_id,
                "bbox_source": annotation.get("bbox_source"),
                "bbox_method": annotation.get("bbox_method"),
            })

        # --------------------------------------------------
        # Verify bbox against keypoints
        # --------------------------------------------------

        keypoints = annotation.get("keypoints", [])

        expected = expected_bbox(keypoints)

        if expected is not None:

            expected_x, expected_y, expected_w, expected_h = expected

            if not all(
                approximately_equal(actual, expected)
                for actual, expected in zip(
                    (x, y, bbox_width, bbox_height),
                    expected,
                )
            ):
                results["bbox_mismatch"].append({
                    "annotation_id": annotation_id,
                    "image_id": image_id,
                    "bbox": bbox,
                    "expected_bbox": list(expected),
                })
                continue

        results["valid"] += 1

    return results


def print_results(results: dict[str, Any]) -> None:

    print()
    print("=" * 64)
    print(results["dataset"])
    print("=" * 64)

    print(f"Images:              {results['images']}")
    print(f"Annotations:         {results['annotations']}")
    print(f"Valid:               {results['valid']}")
    print(
        f"Invalid structure:   "
        f"{len(results['invalid_structure'])}"
    )
    print(
        f"Non-positive:        "
        f"{len(results['non_positive'])}"
    )
    print(
        f"Out of bounds:       "
        f"{len(results['out_of_bounds'])}"
    )
    print(
        f"BBox mismatch:       "
        f"{len(results['bbox_mismatch'])}"
    )
    print(
        f"Missing metadata:    "
        f"{len(results['missing_metadata'])}"
    )

    if results["out_of_bounds"]:
        print()
        print("OUT-OF-BOUNDS ANNOTATIONS:")

        for item in results["out_of_bounds"]:
            print(
                f"  annotation_id={item['annotation_id']} "
                f"image_id={item['image_id']} "
                f"file={item['file_name']}"
            )
            print(
                f"    image: {item['image_width']} x "
                f"{item['image_height']}"
            )
            print(
                f"    bbox: {item['bbox']}"
            )
            print(
                f"    xyxy: {item['bbox_xyxy']}"
            )

    if results["bbox_mismatch"]:
        print()
        print("BBOX MISMATCHES:")

        for item in results["bbox_mismatch"][:10]:
            print(
                f"  annotation_id={item['annotation_id']} "
                f"image_id={item['image_id']}"
            )
            print(
                f"    generated: {item['bbox']}"
            )
            print(
                f"    expected:  {item['expected_bbox']}"
            )

    print()


def main() -> None:

    print("=" * 64)
    print("B4 KEYPOINT-DERIVED BBOX VALIDATION")
    print("=" * 64)

    all_results = []

    for dataset_name, path in INPUT_FILES.items():

        if not path.exists():
            raise FileNotFoundError(
                f"Input file not found: {path}"
            )

        results = validate_dataset(
            dataset_name,
            path,
        )

        all_results.append(results)

        print_results(results)

    print("=" * 64)
    print("VALIDATION COMPLETE")
    print("=" * 64)

    total_annotations = sum(
        result["annotations"]
        for result in all_results
    )

    total_valid = sum(
        result["valid"]
        for result in all_results
    )

    total_out_of_bounds = sum(
        len(result["out_of_bounds"])
        for result in all_results
    )

    total_mismatches = sum(
        len(result["bbox_mismatch"])
        for result in all_results
    )

    print(f"Total annotations:  {total_annotations}")
    print(f"Total valid:        {total_valid}")
    print(f"Out of bounds:      {total_out_of_bounds}")
    print(f"BBox mismatches:    {total_mismatches}")


if __name__ == "__main__":
    main()