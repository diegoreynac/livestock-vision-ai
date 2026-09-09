from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


PROJECT_ROOT = Path(__file__).resolve().parents[1]

JSON_PATH = (
    PROJECT_ROOT
    / "output"
    / "keypoint_bbox_annotations"
    / "coco_b4_rear_keypoint_bbox.json"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "output"
    / "b4_keypoint_bbox_visualization"
)

TARGET_ANNOTATION_ID = 363
TARGET_IMAGE_ID = 2304


def load_data() -> dict:
    with JSON_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def find_target(data: dict) -> tuple[dict, dict]:
    image = next(
        image
        for image in data["images"]
        if image["id"] == TARGET_IMAGE_ID
    )

    annotation = next(
        annotation
        for annotation in data["annotations"]
        if annotation["id"] == TARGET_ANNOTATION_ID
    )

    return image, annotation


def extract_keypoints(annotation: dict) -> list[tuple[float, float]]:
    keypoints = annotation.get("keypoints", [])

    if len(keypoints) % 3 != 0:
        raise ValueError("Invalid COCO keypoints format.")

    points = []

    for index in range(0, len(keypoints), 3):
        x = float(keypoints[index])
        y = float(keypoints[index + 1])
        visibility = float(keypoints[index + 2])

        if visibility > 0:
            points.append((x, y))

    return points


def generate_visualization(image: dict, annotation: dict) -> Path:
    width = float(image["width"])
    height = float(image["height"])

    bbox = annotation["bbox"]

    x, y, bbox_width, bbox_height = map(float, bbox)

    keypoints = extract_keypoints(annotation)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_path = (
        OUTPUT_DIR
        / "b4_rear_out_of_bounds_182_b4-4_r_155_F.png"
    )

    fig, ax = plt.subplots(figsize=(12, 8))

    # Blank image canvas.
    ax.set_xlim(0, width)
    ax.set_ylim(height, 0)
    ax.set_aspect("equal")

    # Image boundary.
    boundary = Rectangle(
        (0, 0),
        width,
        height,
        fill=False,
        edgecolor="black",
        linestyle="--",
        linewidth=2,
        label="Image boundary",
    )

    ax.add_patch(boundary)

    # Derived BBox.
    bbox_patch = Rectangle(
        (x, y),
        bbox_width,
        bbox_height,
        fill=False,
        edgecolor="red",
        linewidth=3,
        label="Derived candidate bbox",
    )

    ax.add_patch(bbox_patch)

    # Keypoints.
    for index, (point_x, point_y) in enumerate(keypoints):
        ax.scatter(
            point_x,
            point_y,
            s=100,
            color="lime",
            edgecolors="black",
            linewidths=1,
            zorder=3,
        )

        ax.text(
            point_x + 12,
            point_y - 12,
            str(index),
            fontsize=12,
            color="green",
            weight="bold",
            zorder=4,
        )

    # Identify the keypoint responsible for the minimum Y.
    min_y_index, (min_x, min_y) = min(
        enumerate(keypoints),
        key=lambda item: item[1][1],
    )

    ax.scatter(
        min_x,
        min_y,
        s=180,
        facecolors="none",
        edgecolors="blue",
        linewidths=2,
        zorder=5,
    )

    ax.text(
        min_x + 20,
        min_y + 30,
        f"min y = {min_y:.0f}",
        fontsize=11,
        color="blue",
        weight="bold",
        zorder=5,
    )

    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    ax.set_title(
        "B4 Rear | "
        f"annotation_id={annotation['id']} | "
        f"image_id={image['id']} | "
        f"{image['file_name']}\n"
        f"size={int(width)}x{int(height)} | "
        f"keypoints_used={len(keypoints)} | "
        f"derived bbox={tuple(map(int, bbox))}"
    )

    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    return output_path


def main() -> None:
    print("=" * 64)
    print("B4 REAR OUT-OF-BOUNDS BBOX VISUALIZATION")
    print("=" * 64)
    print()

    data = load_data()
    image, annotation = find_target(data)

    output_path = generate_visualization(image, annotation)

    bbox = annotation["bbox"]
    x, y, bbox_width, bbox_height = bbox

    print(f"Image:          {image['file_name']}")
    print(f"Image ID:       {image['id']}")
    print(f"Annotation ID:  {annotation['id']}")
    print(
        f"Image size:     "
        f"{image['width']} x {image['height']}"
    )
    print(f"BBox:           {bbox}")
    print(
        f"XYXY:           "
        f"({x}, {y}, {x + bbox_width}, {y + bbox_height})"
    )
    print()
    print(f"Output:         {output_path}")
    print()
    print("=" * 64)
    print("VISUALIZATION COMPLETE")
    print("=" * 64)


if __name__ == "__main__":
    main()