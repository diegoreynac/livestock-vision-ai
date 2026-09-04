import unittest
import torch
from pathlib import Path
import tempfile

from src.models.torch_models import DualViewTorchModel
from src.models.output import ModelOutput
from src.models.base import BaseModel
from src.training.torch_dataset import InputMode


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

    def test_forward_uses_legacy_bbox_until_per_view_migration(self):
        # Transitional contract: the model has not migrated to per-view boxes
        # yet, so SIDE_REAR populates the legacy bbox/sex/weight fields and
        # leaves bbox_side/bbox_rear unset. Update this test when the model
        # adopts the per-view output contract.
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        out = model(self.side, self.rear)
        self.assertIsInstance(out.bbox, torch.Tensor)
        self.assertIsNone(out.bbox_side)
        self.assertIsNone(out.bbox_rear)

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


class TestInputModeContract(unittest.TestCase):
    """Input-mode contract tests for DualViewTorchModel.

    The single-view (SIDE/REAR) architecture is intentionally not implemented
    yet; these tests pin the input contract only.
    """

    def setUp(self):
        self.side = torch.randn(1, 3, 224, 224)
        self.rear = torch.randn(1, 3, 224, 224)

    def _make_model(self, input_mode):
        return DualViewTorchModel(
            architecture="mobilenet",
            variant="small",
            share_backbone=True,
            input_mode=input_mode,
        )

    def test_each_input_mode_can_be_constructed(self):
        for mode in (InputMode.SIDE, InputMode.REAR, InputMode.SIDE_REAR):
            with self.subTest(input_mode=mode):
                model = self._make_model(mode)
                self.assertIs(model.input_mode, mode)

    def test_default_input_mode_is_side_rear(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        self.assertIs(model.input_mode, InputMode.SIDE_REAR)

    def test_invalid_input_mode_is_rejected(self):
        for bad_mode in ("side", "side_rear", None, 0, ["side"]):
            with self.subTest(input_mode=bad_mode):
                with self.assertRaises(TypeError):
                    self._make_model(bad_mode)

    def test_side_mode_accepts_one_image_but_architecture_pending(self):
        model = self._make_model(InputMode.SIDE)
        with self.assertRaisesRegex(NotImplementedError, "single-view"):
            model(self.side)

    def test_side_mode_rejects_wrong_input_contract(self):
        model = self._make_model(InputMode.SIDE)
        with self.assertRaises(TypeError):
            model()  # zero inputs
        with self.assertRaises(TypeError):
            model(self.side, self.rear)  # two inputs
        with self.assertRaises(TypeError):
            model("not-a-tensor")  # single input, wrong type

    def test_rear_mode_accepts_one_image_but_architecture_pending(self):
        model = self._make_model(InputMode.REAR)
        with self.assertRaisesRegex(NotImplementedError, "single-view"):
            model(self.rear)  # single positional tensor
        with self.assertRaisesRegex(NotImplementedError, "single-view"):
            model(rear=self.rear)  # keyword form

    def test_rear_mode_rejects_wrong_input_contract(self):
        model = self._make_model(InputMode.REAR)
        with self.assertRaises(TypeError):
            model()  # zero inputs
        with self.assertRaises(TypeError):
            model(self.side, self.rear)  # two inputs
        with self.assertRaises(TypeError):
            model(self.rear, unexpected=True)  # unexpected kwarg

    def test_side_rear_accepts_side_and_rear(self):
        model = self._make_model(InputMode.SIDE_REAR)
        out = model(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)

    def test_side_rear_rejects_wrong_input_contract(self):
        model = self._make_model(InputMode.SIDE_REAR)
        with self.assertRaises(TypeError):
            model(self.side)  # missing rear view
        with self.assertRaises(TypeError):
            model(rear=self.rear)  # missing side view
        with self.assertRaises(TypeError):
            model(self.side, self.rear, unexpected=True)  # unexpected kwarg
        with self.assertRaises(TypeError):
            model(self.side, "not-a-tensor")  # wrong type

    def test_side_rear_forward_output_shapes(self):
        model = self._make_model(InputMode.SIDE_REAR)
        side = torch.randn(2, 3, 224, 224)
        rear = torch.randn(2, 3, 224, 224)
        out = model(side, rear)
        self.assertEqual(tuple(out.bbox.shape), (2, 4))
        self.assertEqual(tuple(out.sex.shape), (2, 2))
        self.assertEqual(tuple(out.weight.shape), (2, 1))

    def test_forward_remains_differentiable(self):
        model = self._make_model(InputMode.SIDE_REAR)
        model.train()
        side = self.side.clone().requires_grad_(True)
        rear = self.rear.clone().requires_grad_(True)
        out = model(side, rear)
        loss = out.bbox.sum() + out.sex.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(side.grad)
        self.assertIsNotNone(rear.grad)
        self.assertIsNotNone(model.fusion[0].weight.grad)
        self.assertTrue(torch.isfinite(model.fusion[0].weight.grad).all())


if __name__ == "__main__":
    unittest.main()
