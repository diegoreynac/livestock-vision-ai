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

    def test_yolo26_nano_forward_dual_view(self):
        for share_backbone in (True, False):
            with self.subTest(share_backbone=share_backbone):
                model = DualViewTorchModel(architecture="yolo", variant="nano", share_backbone=share_backbone)
                out = model.predict(self.side, self.rear)
                self.assertIsInstance(out, ModelOutput)
                self.assertIsInstance(out.bbox, tuple)
                self.assertIsInstance(out.sex, str)
                self.assertIsInstance(out.weight, float)
                self.assertGreater(model.count_parameters(), 0)
                self.assertGreater(model.model_size(), 0.0)

    def test_yolo26_supported_variants_and_dual_view(self):
        for variant in ("nano", "small", "medium"):
            with self.subTest(variant=variant):
                model = DualViewTorchModel(architecture="yolo", variant=variant, share_backbone=False)
                out = model.predict(self.side, self.rear)
                self.assertIsInstance(out, ModelOutput)
                self.assertIsInstance(out.bbox, tuple)
                self.assertIsInstance(out.sex, str)
                self.assertIsInstance(out.weight, float)

    def test_yolo_unsupported_variant_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO variant"):
            DualViewTorchModel(architecture="yolo", variant="unsupported")

    def test_forward_returns_torch_tensors(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        out = model(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox, torch.Tensor)
        self.assertIsInstance(out.sex, torch.Tensor)
        self.assertIsInstance(out.weight, torch.Tensor)
        self.assertEqual(tuple(out.bbox.shape), (1, 4))
        self.assertEqual(tuple(out.sex.shape), (1, 2))
        self.assertEqual(tuple(out.weight.shape), (1, 1))

    def test_forward_outputs_require_grad(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        side = self.side.clone().requires_grad_(True)
        rear = self.rear.clone().requires_grad_(True)
        out = model(side, rear)
        self.assertTrue(out.bbox.requires_grad)
        self.assertTrue(out.sex.requires_grad)
        self.assertTrue(out.weight.requires_grad)
        self.assertIsNotNone(out.bbox.grad_fn)
        self.assertIsNotNone(out.weight.grad_fn)

    def test_scalar_loss_backward(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        model.train()
        out = model(self.side, self.rear)
        loss = out.bbox.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(model.bbox_head.weight.grad)
        self.assertIsNotNone(model.weight_head.weight.grad)
        self.assertTrue(torch.isfinite(model.bbox_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.weight_head.weight.grad).all())

    def test_predict_returns_python_values_single_and_batch(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)

        out_single = model.predict(self.side, self.rear)
        self.assertIsInstance(out_single.bbox, tuple)
        self.assertEqual(len(out_single.bbox), 4)
        self.assertTrue(all(isinstance(v, float) for v in out_single.bbox))
        self.assertIn(out_single.sex, ("F", "M"))
        self.assertIsInstance(out_single.weight, float)

        side_batch = torch.randn(2, 3, 224, 224)
        rear_batch = torch.randn(2, 3, 224, 224)
        out_batch = model.predict(side_batch, rear_batch)
        self.assertIsInstance(out_batch.bbox, list)
        self.assertEqual(len(out_batch.bbox), 2)
        self.assertTrue(all(isinstance(row, tuple) and len(row) == 4 for row in out_batch.bbox))
        self.assertIsInstance(out_batch.sex, list)
        self.assertEqual(len(out_batch.sex), 2)
        self.assertTrue(all(label in ("F", "M") for label in out_batch.sex))
        self.assertIsInstance(out_batch.weight, list)
        self.assertEqual(len(out_batch.weight), 2)
        self.assertTrue(all(isinstance(w, float) for w in out_batch.weight))

    def test_predict_restores_training_mode(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)

        model.train()
        model.predict(self.side, self.rear)
        self.assertTrue(model.training)

        model.eval()
        model.predict(self.side, self.rear)
        self.assertFalse(model.training)


if __name__ == "__main__":
    unittest.main()
