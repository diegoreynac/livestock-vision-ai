from __future__ import annotations

import sys
from pathlib import Path

# Allow direct execution from the repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.analysis.keypoint_bbox_generation import KeypointBBoxGenerator


DATASET_ROOT = PROJECT_ROOT

OUTPUT_ROOT = PROJECT_ROOT / "output" / "keypoint_bbox_annotations"

DATASET_SOURCES = {
    "B4 Side": DATASET_ROOT / "coco_b4_side.json",
    "B4 Rear": DATASET_ROOT / "coco_b4_rear.json",
}


def main() -> None:
    generator = KeypointBBoxGenerator()

    print("=" * 60)
    print("B4 KEYPOINT-DERIVED BBOX GENERATION")
    print("=" * 60)

    total_replaced = 0

    for name, source in DATASET_SOURCES.items():
        if not source.exists():
            raise FileNotFoundError(
                f"Source COCO file not found for {name}: {source}"
            )

        output_name = source.stem + "_keypoint_bbox.json"
        destination = OUTPUT_ROOT / output_name

        replaced_count = generator.generate_file(
            source,
            destination,
        )

        total_replaced += replaced_count

        print()
        print(name)
        print("-" * 60)
        print(f"Source:             {source}")
        print(f"Destination:        {destination}")
        print(f"Replaced BBoxes:    {replaced_count}")

    print()
    print("=" * 60)
    print(f"TOTAL BBOXES GENERATED: {total_replaced}")
    print("=" * 60)


if __name__ == "__main__":
    main()