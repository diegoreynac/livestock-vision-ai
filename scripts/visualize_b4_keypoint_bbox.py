"""Read-only visualization of B4 keypoints and derived XYXY bboxes.

This script is a diagnostic/investigative tool only. It:
    * reads ``coco_b4_side.json`` / ``coco_b4_rear.json`` from the repository
      root (no other files are touched),
    * reuses the existing keypoint parsing / candidate bbox derivation logic
      from :mod:`src.analysis.bbox_annotation_audit` (no duplicated logic),
    * draws, per sampled annotation: the keypoints, the derived candidate
      XYXY bounding box, and the image boundary (as reported by the COCO
      ``images[]`` entry),
    * saves the figures only under ``output/`` (gitignored).

It does NOT require the actual image files to exist -- since B4 bbox values
are always ``[0, 0, 0, 0]`` and the task only concerns keypoint-derived
geometry, a blank canvas sized to the COCO ``width``/``height`` is used as
the drawing surface. If a real image file is found next to the JSON (or in a
configured images directory), it is used as the background instead.

This script does not modify any source JSON, does not modify project code,
and does not delete/repair any annotations.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

from src.analysis.bbox_annotation_audit import (
    BBoxAnnotationAudit,
    derive_candidate_bbox_from_keypoints,
)

SIDE_JSON = REPO_ROOT / "coco_b4_side.json"
REAR_JSON = REPO_ROOT / "coco_b4_rear.json"
OUTPUT_DIR = REPO_ROOT / "output" / "b4_keypoint_bbox_visualization"
SAMPLES_PER_VIEW = 3


def _load_coco(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _select_sample_annotations(
    coco: dict[str, Any],
    *,
    count: int,
) -> list[dict[str, Any]]:
    """Deterministically pick the first ``count`` annotations that have
    enough usable keypoints to derive a candidate bbox."""

    images_by_id = {image["id"]: image for image in coco.get("images", [])}
    selected: list[dict[str, Any]] = []

    for annotation in coco.get("annotations", []):
        image = images_by_id.get(annotation.get("image_id"))
        if image is None:
            continue

        keypoint_entries = BBoxAnnotationAudit._parse_keypoints(
            annotation.get("keypoints", [])
        )
        candidate = derive_candidate_bbox_from_keypoints(keypoint_entries)
        if candidate is None:
            continue

        selected.append(
            {
                "annotation": annotation,
                "image": image,
                "keypoints": keypoint_entries,
                "candidate": candidate,
            }
        )

        if len(selected) >= count:
            break

    return selected


def _find_real_image(file_name: str) -> Path | None:
    """Best-effort lookup of a real image file; visualization does not
    require this to succeed."""

    for candidate_dir in (REPO_ROOT, REPO_ROOT / "images", REPO_ROOT / "data"):
        candidate_path = candidate_dir / file_name
        if candidate_path.is_file():
            return candidate_path
    return None


def _render_sample(entry: dict[str, Any], *, view_label: str, output_path: Path) -> None:
    image_meta = entry["image"]
    annotation = entry["annotation"]
    keypoints = entry["keypoints"]
    candidate = entry["candidate"]

    width = int(image_meta["width"])
    height = int(image_meta["height"])
    file_name = image_meta.get("file_name", f"image_{image_meta.get('id')}")

    real_image_path = _find_real_image(file_name)

    fig, ax = plt.subplots(figsize=(8, 8 * height / width if width else 8))

    if real_image_path is not None:
        background = plt.imread(real_image_path)
        ax.imshow(background)
    else:
        # No real image available -- draw a neutral canvas sized to the
        # COCO width/height so geometry (keypoints/bbox/boundary) is still
        # shown to correct scale.
        ax.imshow(np.full((height, width, 3), 235, dtype=np.uint8))

    # Image boundary (as reported by COCO images[]).
    boundary = patches.Rectangle(
        (0, 0),
        width,
        height,
        linewidth=2,
        edgecolor="black",
        facecolor="none",
        linestyle="--",
        label="Image boundary",
    )
    ax.add_patch(boundary)

    # Derived candidate XYXY bbox.
    candidate_rect = patches.Rectangle(
        (candidate.x_min, candidate.y_min),
        candidate.width,
        candidate.height,
        linewidth=2,
        edgecolor="red",
        facecolor="none",
        label="Derived candidate bbox (from keypoints)",
    )
    ax.add_patch(candidate_rect)

    # Keypoints, colored by label/finiteness status.
    for point in keypoints:
        if point.x is None or point.y is None:
            continue
        if point.is_finite and point.is_labeled:
            color = "lime"
        elif point.is_finite:
            color = "orange"
        else:
            color = "gray"
        ax.plot(point.x, point.y, marker="o", markersize=6, color=color)
        ax.annotate(
            str(point.index),
            (point.x, point.y),
            textcoords="offset points",
            xytext=(4, 4),
            fontsize=7,
            color=color,
        )

    ax.set_xlim(-0.05 * width, 1.05 * width)
    ax.set_ylim(1.05 * height, -0.05 * height)
    ax.set_title(
        f"{view_label} | annotation_id={annotation.get('id')} | "
        f"image_id={image_meta.get('id')} | {file_name}\n"
        f"size={width}x{height} | keypoints_used={candidate.source_keypoint_count} | "
        f"derived bbox={candidate.as_xywh}"
    )
    ax.legend(loc="upper right", fontsize=7)
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)


def _visualize_view(json_path: Path, *, view_label: str, output_dir: Path) -> list[Path]:
    coco = _load_coco(json_path)
    samples = _select_sample_annotations(coco, count=SAMPLES_PER_VIEW)

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for position, entry in enumerate(samples, start=1):
        output_path = output_dir / f"b4_{view_label.lower()}_sample_{position}.png"
        _render_sample(entry, view_label=view_label, output_path=output_path)
        written.append(output_path)

    return written


def main() -> None:
    written_paths: list[Path] = []
    written_paths += _visualize_view(SIDE_JSON, view_label="Side", output_dir=OUTPUT_DIR)
    written_paths += _visualize_view(REAR_JSON, view_label="Rear", output_dir=OUTPUT_DIR)

    print(f"Wrote {len(written_paths)} visualization(s) under {OUTPUT_DIR}:")
    for path in written_paths:
        print(f"  - {path}")


if __name__ == "__main__":
    main()
