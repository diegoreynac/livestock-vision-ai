import tempfile
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
