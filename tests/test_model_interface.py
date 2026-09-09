import tempfile
import unittest
from pathlib import Path

import torch

from src.models.output import ModelOutput
from src.models.base import BaseModel


class DummyModel(BaseModel):
    """Minimal concrete implementation used only for interface tests."""

    def forward(self, *inputs, **kwargs) -> ModelOutput:
        return ModelOutput(bbox=(0, 0, 1, 1), sex="M", weight=100.0)

    def predict(self, *inputs, **kwargs) -> ModelOutput:
        # For the interface test a predict can delegate to forward
        return self.forward(*inputs, **kwargs)

    def count_parameters(self) -> int:
        return 123

    def model_size(self) -> float:
        return 12.34

    def export(self, destination: Path | str, **kwargs) -> None:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "export_stub.txt").write_text("exported", encoding="utf-8")


class TestModelInterface(unittest.TestCase):
    def test_model_output_construction_and_preservation(self):
        out = ModelOutput(bbox=(1, 2, 3, 4), sex="F", weight=250.5)
        self.assertEqual(out.bbox, (1, 2, 3, 4))
        self.assertEqual(out.sex, "F")
        self.assertEqual(out.weight, 250.5)

    def test_base_model_is_abstract(self):
        with self.assertRaises(TypeError):
            BaseModel()

    def test_dummy_model_implements_interface(self):
        model = DummyModel()
        self.assertIsInstance(model, BaseModel)

        fwd = model.forward(None)
        pred = model.predict(None)
        self.assertIsInstance(fwd, ModelOutput)
        self.assertIsInstance(pred, ModelOutput)
        self.assertEqual(fwd.bbox, (0, 0, 1, 1))
        self.assertEqual(pred.sex, "M")

        self.assertIsInstance(model.count_parameters(), int)
        self.assertIsInstance(model.model_size(), float)

        with tempfile.TemporaryDirectory() as tmp:
            model.export(tmp)
            self.assertTrue((Path(tmp) / "export_stub.txt").exists())


class TestModelOutputContract(unittest.TestCase):
    """Tests for the per-view model output contract (bbox_side/bbox_rear)."""

    def test_per_view_fields_populated(self):
        out = ModelOutput(bbox_side=(0, 0, 1, 1), bbox_rear=(2, 2, 3, 3), weight=250.5, sex="F")
        self.assertEqual(out.bbox_side, (0, 0, 1, 1))
        self.assertEqual(out.bbox_rear, (2, 2, 3, 3))
        self.assertEqual(out.weight, 250.5)
        self.assertEqual(out.sex, "F")
        # Legacy single-box field stays unset under the new contract.
        self.assertIsNone(out.bbox)

    def test_all_fields_optional_and_default_to_none(self):
        out = ModelOutput()
        self.assertIsNone(out.bbox_side)
        self.assertIsNone(out.bbox_rear)
        self.assertIsNone(out.weight)
        self.assertIsNone(out.sex)
        self.assertIsNone(out.bbox)

    def test_side_view_only_output(self):
        out = ModelOutput(bbox_side=(0, 0, 1, 1), weight=100.0)
        self.assertEqual(out.bbox_side, (0, 0, 1, 1))
        self.assertIsNone(out.bbox_rear)
        self.assertIsNone(out.sex)

    def test_rear_view_only_output(self):
        out = ModelOutput(bbox_rear=(0, 0, 1, 1), weight=100.0)
        self.assertEqual(out.bbox_rear, (0, 0, 1, 1))
        self.assertIsNone(out.bbox_side)
        self.assertIsNone(out.sex)

    def test_fields_accept_torch_tensors(self):
        bbox_side = torch.randn(2, 4)
        bbox_rear = torch.randn(2, 4)
        weight = torch.randn(2, 1)
        sex = torch.randn(2, 2)
        out = ModelOutput(bbox_side=bbox_side, bbox_rear=bbox_rear, weight=weight, sex=sex)
        self.assertIs(out.bbox_side, bbox_side)
        self.assertIs(out.bbox_rear, bbox_rear)
        self.assertIs(out.weight, weight)
        self.assertIs(out.sex, sex)

    def test_legacy_bbox_field_remains_supported(self):
        out = ModelOutput(bbox=(1, 2, 3, 4), sex="M", weight=100.0)
        self.assertEqual(out.bbox, (1, 2, 3, 4))
        self.assertIsNone(out.bbox_side)
        self.assertIsNone(out.bbox_rear)

    def test_dataclass_equality_semantics(self):
        first = ModelOutput(bbox_side=(1, 2, 3, 4), weight=1.0)
        second = ModelOutput(bbox_side=(1, 2, 3, 4), weight=1.0)
        self.assertEqual(first, second)
        self.assertNotEqual(first, ModelOutput(bbox_rear=(1, 2, 3, 4), weight=1.0))


if __name__ == "__main__":
    unittest.main()
