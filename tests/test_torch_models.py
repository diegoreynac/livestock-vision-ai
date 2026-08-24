import unittest
import torch
from pathlib import Path
import tempfile

from src.models.torch_models import DualViewTorchModel
from src.models.output import ModelOutput
from src.models.base import BaseModel


class TestTorchModels(unittest.TestCase):
    def setUp(self):
        # Small random tensors simulating RGB images 224x224
        self.side = torch.randn(1, 3, 224, 224)
        self.rear = torch.randn(1, 3, 224, 224)

    def test_mobilenet_forward_and_interface(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=False)
        self.assertIsInstance(model, BaseModel)
        out = model.predict(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox, tuple)
        self.assertIsInstance(out.sex, str)
        self.assertIsInstance(out.weight, float)
        # parameter counting and size
        params = model.count_parameters()
        size_mb = model.model_size()
        self.assertIsInstance(params, int)
        self.assertGreater(params, 0)
        self.assertIsInstance(size_mb, float)
        self.assertGreater(size_mb, 0.0)
        with tempfile.TemporaryDirectory() as tmp:
            model.export(tmp)
            self.assertTrue((Path(tmp) / "model_state.pth").exists())

    def test_efficientnet_forward(self):
        model = DualViewTorchModel(architecture="efficientnet", variant="b0")
        out = model.predict(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)

    def test_yolo_forward(self):
        model = DualViewTorchModel(architecture="yolo", variant="nano")
        out = model.predict(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)

    def test_yolo_unsupported_variant_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO variant"):
            DualViewTorchModel(architecture="yolo", variant="unsupported")


if __name__ == "__main__":
    unittest.main()
