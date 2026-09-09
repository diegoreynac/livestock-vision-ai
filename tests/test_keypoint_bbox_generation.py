from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.analysis.keypoint_bbox_generation import KeypointBBoxGenerator


class TestKeypointBBoxGenerator:
    @staticmethod
    def _base_coco() -> dict:
        return {
            "info": {
                "description": "Synthetic livestock dataset",
            },
            "licenses": [],
            "images": [
                {
                    "id": 1,
                    "file_name": "animal_001.jpg",
                    "width": 100,
                    "height": 80,
                },
                {
                    "id": 2,
                    "file_name": "animal_002.jpg",
                    "width": 100,
                    "height": 80,
                },
            ],
            "annotations": [
                {
                    "id": 10,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 0, 0],
                    "area": 0,
                    "iscrowd": 0,
                    "keypoints": [
                        10, 20, 2,
                        50, 20, 2,
                        50, 60, 2,
                        10, 60, 2,
                    ],
                    "num_keypoints": 4,
                },
                {
                    "id": 20,
                    "image_id": 2,
                    "category_id": 1,
                    "bbox": [20, 10, 40, 50],
                    "area": 2000,
                    "iscrowd": 0,
                    "keypoints": [
                        20, 10, 2,
                        60, 10, 2,
                        60, 60, 2,
                        20, 60, 2,
                    ],
                    "num_keypoints": 4,
                },
            ],
            "categories": [
                {
                    "id": 1,
                    "name": "cattle",
                }
            ],
        }

    def test_replaces_invalid_bbox_using_keypoints(self):
        data = self._base_coco()

        generated, replaced_count = KeypointBBoxGenerator().generate(data)

        self.assert_annotation_bbox(
            generated,
            annotation_id=10,
            expected_bbox=[10.0, 20.0, 40.0, 40.0],
        )
        assert replaced_count == 1

        annotation = self._annotation(generated, 10)
        assert annotation["bbox_source"] == "keypoints"
        assert annotation["bbox_method"] == "min_max_visible_keypoints"
        assert annotation["area"] == 1600.0
        assert annotation["isbbox"] is True

    def test_preserves_valid_bbox(self):
        data = self._base_coco()

        generated, replaced_count = KeypointBBoxGenerator().generate(data)

        self.assert_annotation_bbox(
            generated,
            annotation_id=20,
            expected_bbox=[20, 10, 40, 50],
        )

        annotation = self._annotation(generated, 20)

        assert "bbox_source" not in annotation
        assert "bbox_method" not in annotation

        assert replaced_count == 1

    def test_does_not_modify_original_data(self):
        data = self._base_coco()

        original_bbox = list(data["annotations"][0]["bbox"])
        original_area = data["annotations"][0]["area"]

        generated, _ = KeypointBBoxGenerator().generate(data)

        assert data["annotations"][0]["bbox"] == original_bbox
        assert data["annotations"][0]["area"] == original_area

        assert generated["annotations"][0]["bbox"] != original_bbox

    def test_preserves_coco_structure_and_other_fields(self):
        data = self._base_coco()

        generated, _ = KeypointBBoxGenerator().generate(data)

        assert generated["info"] == data["info"]
        assert generated["licenses"] == data["licenses"]
        assert generated["images"] == data["images"]
        assert generated["categories"] == data["categories"]

        generated_annotation = self._annotation(generated, 10)
        original_annotation = self._annotation(data, 10)

        assert generated_annotation["image_id"] == original_annotation["image_id"]
        assert generated_annotation["category_id"] == original_annotation["category_id"]
        assert generated_annotation["keypoints"] == original_annotation["keypoints"]
        assert generated_annotation["num_keypoints"] == original_annotation["num_keypoints"]

    def test_does_not_generate_bbox_without_valid_candidate(self):
        data = self._base_coco()

        data["annotations"][0]["keypoints"] = [
            10, 20, 2,
        ]
        data["annotations"][0]["num_keypoints"] = 1

        generated, replaced_count = KeypointBBoxGenerator().generate(data)

        assert replaced_count == 0
        assert generated["annotations"][0]["bbox"] == [0, 0, 0, 0]
        assert "bbox_source" not in generated["annotations"][0]

    def test_ignores_annotation_without_keypoints(self):
        data = self._base_coco()

        data["annotations"][0]["keypoints"] = []
        data["annotations"][0]["num_keypoints"] = 0

        generated, replaced_count = KeypointBBoxGenerator().generate(data)

        assert replaced_count == 0
        assert generated["annotations"][0]["bbox"] == [0, 0, 0, 0]

    def test_generate_file_creates_independent_json(self, tmp_path: Path):
        source = tmp_path / "source.json"
        destination = tmp_path / "generated" / "derived.json"

        data = self._base_coco()

        source.write_text(
            json.dumps(data),
            encoding="utf-8",
        )

        replaced_count = KeypointBBoxGenerator().generate_file(
            source,
            destination,
        )

        assert replaced_count == 1
        assert destination.exists()

        generated = json.loads(
            destination.read_text(encoding="utf-8")
        )

        self.assert_annotation_bbox(
            generated,
            annotation_id=10,
            expected_bbox=[10.0, 20.0, 40.0, 40.0],
        )

        original = json.loads(
            source.read_text(encoding="utf-8")
        )

        assert original["annotations"][0]["bbox"] == [0, 0, 0, 0]

    def test_rejects_invalid_coco_data(self):
        generator = KeypointBBoxGenerator()

        with pytest.raises(TypeError):
            generator.generate(None)

        with pytest.raises(ValueError):
            generator.generate({})

        with pytest.raises(ValueError):
            generator.generate({"images": []})

        with pytest.raises(ValueError):
            generator.generate(
                {
                    "images": [],
                    "annotations": {},
                }
            )

    @staticmethod
    def _annotation(data: dict, annotation_id: int) -> dict:
        for annotation in data["annotations"]:
            if annotation["id"] == annotation_id:
                return annotation

        raise AssertionError(
            f"Annotation {annotation_id} was not found."
        )

    def assert_annotation_bbox(
        self,
        data: dict,
        *,
        annotation_id: int,
        expected_bbox: list[float],
    ) -> None:
        annotation = self._annotation(data, annotation_id)

        assert annotation["bbox"] == pytest.approx(expected_bbox)