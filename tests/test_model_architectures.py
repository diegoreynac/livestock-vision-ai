import tempfile
import unittest
from pathlib import Path

from src.models.dual_view_model import DualViewModel
from src.models.architectures import mobilenet, efficientnet, yolo
from src.models.base import BaseModel
from src.models.output import ModelOutput


class TestModelArchitectures(unittest.TestCase):
    def test_dual_view_dummy_forward_and_interface(self):
        # Test a representative set of architectures and variants
        combos = [
            ("mobilenet", "default"),
            ("mobilenet", "small"),
            ("efficientnet", "default"),
            ("efficientnet", "small"),
            ("yolo", "nano"),
            ("yolo", "small"),
        ]

        for arch, variant in combos:
            with self.subTest(arch=arch, variant=variant):
                model = DualViewModel(architecture=arch, variant=variant)
                # Ensure BaseModel contract
                self.assertIsInstance(model, BaseModel)

                out = model.forward(None, None)
                self.assertIsInstance(out, ModelOutput)
                # expected fields
                self.assertTrue(hasattr(out, "bbox"))
                self.assertTrue(hasattr(out, "sex"))
                self.assertTrue(hasattr(out, "weight"))

                # Parameter count and model size sanity checks
                params = model.count_parameters()
                size_mb = model.model_size()
                self.assertIsInstance(params, int)
                self.assertGreater(params, 0)
                self.assertIsInstance(size_mb, float)
                self.assertGreater(size_mb, 0.0)

                # Export writes metadata
                with tempfile.TemporaryDirectory() as tmp:
                    model.export(tmp)
                    self.assertTrue((Path(tmp) / "model_metadata.txt").exists())

    def test_architecture_factories_are_consistent(self):
        # Ensure the factories produce ArchitectureSpec-like behaviour
        m = mobilenet("small")
        e = efficientnet("b0")
        y = yolo("small")

        self.assertLess(m.count_parameters(), e.count_parameters() + 1000000)  # coarse sanity
        self.assertLess(y.count_parameters(), e.count_parameters() + 1000000)


if __name__ == "__main__":
    unittest.main()
