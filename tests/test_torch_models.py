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
        self.assertIsInstance(out.bbox_side, tuple)
        self.assertIsInstance(out.bbox_rear, tuple)
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
        self.assertIsInstance(out.bbox_side, tuple)
        self.assertIsInstance(out.bbox_rear, tuple)
        self.assertIsInstance(out.weight, float)

    def test_yolo26_nano_forward_dual_view(self):
        for share_backbone in (True, False):
            with self.subTest(share_backbone=share_backbone):
                model = DualViewTorchModel(architecture="yolo", variant="nano", share_backbone=share_backbone)
                out = model.predict(self.side, self.rear)
                self.assertIsInstance(out, ModelOutput)
                self.assertIsInstance(out.bbox_side, tuple)
                self.assertIsInstance(out.bbox_rear, tuple)
                self.assertIsInstance(out.weight, float)
                self.assertGreater(model.count_parameters(), 0)
                self.assertGreater(model.model_size(), 0.0)

    def test_yolo26_supported_variants_and_dual_view(self):
        for variant in ("nano", "small", "medium"):
            with self.subTest(variant=variant):
                model = DualViewTorchModel(architecture="yolo", variant=variant, share_backbone=False)
                out = model.predict(self.side, self.rear)
                self.assertIsInstance(out, ModelOutput)
                self.assertIsInstance(out.bbox_side, tuple)
                self.assertIsInstance(out.bbox_rear, tuple)
                self.assertIsInstance(out.weight, float)

    def test_yolo_unsupported_variant_raises_value_error(self):
        with self.assertRaisesRegex(ValueError, "Unsupported YOLO variant"):
            DualViewTorchModel(architecture="yolo", variant="unsupported")

    def test_forward_returns_torch_tensors(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        out = model(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox_side, torch.Tensor)
        self.assertIsInstance(out.bbox_rear, torch.Tensor)
        self.assertIsInstance(out.sex, torch.Tensor)
        self.assertIsInstance(out.weight, torch.Tensor)
        self.assertEqual(tuple(out.bbox_side.shape), (1, 4))
        self.assertEqual(tuple(out.bbox_rear.shape), (1, 4))
        self.assertEqual(tuple(out.sex.shape), (1, 2))
        self.assertEqual(tuple(out.weight.shape), (1, 1))
        self.assertIsNone(out.bbox)

    def test_forward_outputs_require_grad(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        side = self.side.clone().requires_grad_(True)
        rear = self.rear.clone().requires_grad_(True)
        out = model(side, rear)
        self.assertTrue(out.bbox_side.requires_grad)
        self.assertTrue(out.bbox_rear.requires_grad)
        self.assertTrue(out.sex.requires_grad)
        self.assertTrue(out.weight.requires_grad)
        self.assertIsNotNone(out.bbox_side.grad_fn)
        self.assertIsNotNone(out.bbox_rear.grad_fn)
        self.assertIsNotNone(out.weight.grad_fn)

    def test_scalar_loss_backward(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)
        model.train()
        out = model(self.side, self.rear)
        loss = out.bbox_side.sum() + out.bbox_rear.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(model.bbox_side_head.weight.grad)
        self.assertIsNotNone(model.bbox_rear_head.weight.grad)
        self.assertIsNotNone(model.weight_head.weight.grad)
        self.assertTrue(torch.isfinite(model.bbox_side_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.bbox_rear_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.weight_head.weight.grad).all())

    def test_predict_returns_python_values_single_and_batch(self):
        model = DualViewTorchModel(architecture="mobilenet", variant="small", share_backbone=True)

        out_single = model.predict(self.side, self.rear)
        for bbox in (out_single.bbox_side, out_single.bbox_rear):
            self.assertIsInstance(bbox, tuple)
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(v, float) for v in bbox))
        self.assertIsInstance(out_single.weight, float)
        self.assertIsNone(out_single.bbox)

        side_batch = torch.randn(2, 3, 224, 224)
        rear_batch = torch.randn(2, 3, 224, 224)
        out_batch = model.predict(side_batch, rear_batch)
        for bbox in (out_batch.bbox_side, out_batch.bbox_rear):
            self.assertIsInstance(bbox, list)
            self.assertEqual(len(bbox), 2)
            self.assertTrue(all(isinstance(row, tuple) and len(row) == 4 for row in bbox))
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
    """Input-mode contract tests for DualViewTorchModel."""

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

    def test_side_mode_rejects_wrong_input_contract(self):
        model = self._make_model(InputMode.SIDE)
        with self.assertRaises(TypeError):
            model()  # zero inputs
        with self.assertRaises(TypeError):
            model(self.side, self.rear)  # two inputs
        with self.assertRaises(TypeError):
            model("not-a-tensor")  # single input, wrong type

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


class TestSingleViewSide(unittest.TestCase):
    """InputMode.SIDE: one side image -> bbox_side + weight (+ sex logits)."""

    def setUp(self):
        self.model = DualViewTorchModel(
            architecture="mobilenet",
            variant="small",
            share_backbone=True,
            input_mode=InputMode.SIDE,
        )
        self.side = torch.randn(1, 3, 224, 224)

    def test_single_input_forward_output_contract(self):
        out = self.model(self.side)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox_side, torch.Tensor)
        self.assertEqual(tuple(out.bbox_side.shape), (1, 4))
        self.assertIsNone(out.bbox_rear)
        self.assertIsInstance(out.weight, torch.Tensor)
        self.assertEqual(tuple(out.weight.shape), (1, 1))
        self.assertIsInstance(out.sex, torch.Tensor)
        self.assertEqual(tuple(out.sex.shape), (1, 2))
        self.assertIsNone(out.bbox)

    def test_batch_forward_shapes(self):
        out = self.model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.bbox_side.shape), (2, 4))
        self.assertEqual(tuple(out.weight.shape), (2, 1))
        self.assertEqual(tuple(out.sex.shape), (2, 2))

    def test_forward_outputs_require_grad_and_backward(self):
        self.model.train()
        side = self.side.clone().requires_grad_(True)
        out = self.model(side)
        self.assertTrue(out.bbox_side.requires_grad)
        self.assertTrue(out.weight.requires_grad)
        self.assertIsNotNone(out.bbox_side.grad_fn)
        self.assertIsNotNone(out.weight.grad_fn)

        loss = out.bbox_side.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(side.grad)
        self.assertIsNotNone(self.model.bbox_side_head.weight.grad)
        self.assertIsNotNone(self.model.weight_head.weight.grad)
        # Only the side-view parameters participate in this graph.
        self.assertIsNone(self.model.bbox_rear_head.weight.grad)
        self.assertTrue(torch.isfinite(self.model.bbox_side_head.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.weight_head.weight.grad).all())

    def test_fusion_input_dimension_matches_single_view(self):
        self.assertEqual(self.model.fusion[0].in_features, self.model._backbone_feat_dim)

    def test_predict_returns_python_values_and_restores_mode(self):
        self.model.train()
        out = self.model.predict(self.side)
        self.assertTrue(self.model.training)
        self.assertIsInstance(out.bbox_side, tuple)
        self.assertEqual(len(out.bbox_side), 4)
        self.assertTrue(all(isinstance(v, float) for v in out.bbox_side))
        self.assertIsNone(out.bbox_rear)
        self.assertIsInstance(out.weight, float)
        self.assertIsNone(out.bbox)

        self.model.eval()
        self.model.predict(self.side)
        self.assertFalse(self.model.training)


class TestSingleViewRear(unittest.TestCase):
    """InputMode.REAR: one rear image -> bbox_rear + weight (+ sex logits)."""

    def setUp(self):
        self.model = DualViewTorchModel(
            architecture="mobilenet",
            variant="small",
            share_backbone=True,
            input_mode=InputMode.REAR,
        )
        self.rear = torch.randn(1, 3, 224, 224)

    def test_single_input_forward_output_contract(self):
        out = self.model(self.rear)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox_rear, torch.Tensor)
        self.assertEqual(tuple(out.bbox_rear.shape), (1, 4))
        self.assertIsNone(out.bbox_side)
        self.assertIsInstance(out.weight, torch.Tensor)
        self.assertEqual(tuple(out.weight.shape), (1, 1))
        self.assertIsInstance(out.sex, torch.Tensor)
        self.assertEqual(tuple(out.sex.shape), (1, 2))
        self.assertIsNone(out.bbox)

    def test_keyword_input_forward(self):
        out = self.model(rear=self.rear)
        self.assertIsInstance(out.bbox_rear, torch.Tensor)
        self.assertIsNone(out.bbox_side)

    def test_batch_forward_shapes(self):
        out = self.model(torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.bbox_rear.shape), (2, 4))
        self.assertEqual(tuple(out.weight.shape), (2, 1))
        self.assertEqual(tuple(out.sex.shape), (2, 2))

    def test_forward_outputs_require_grad_and_backward(self):
        self.model.train()
        rear = self.rear.clone().requires_grad_(True)
        out = self.model(rear)
        self.assertTrue(out.bbox_rear.requires_grad)
        self.assertTrue(out.weight.requires_grad)
        self.assertIsNotNone(out.bbox_rear.grad_fn)
        self.assertIsNotNone(out.weight.grad_fn)

        loss = out.bbox_rear.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(rear.grad)
        self.assertIsNotNone(self.model.bbox_rear_head.weight.grad)
        self.assertIsNotNone(self.model.weight_head.weight.grad)
        # Only the rear-view parameters participate in this graph.
        self.assertIsNone(self.model.bbox_side_head.weight.grad)
        self.assertTrue(torch.isfinite(self.model.bbox_rear_head.weight.grad).all())
        self.assertTrue(torch.isfinite(self.model.weight_head.weight.grad).all())

    def test_fusion_input_dimension_matches_single_view(self):
        self.assertEqual(self.model.fusion[0].in_features, self.model._backbone_feat_dim)

    def test_predict_returns_python_values_and_restores_mode(self):
        self.model.train()
        out = self.model.predict(self.rear)
        self.assertTrue(self.model.training)
        self.assertIsInstance(out.bbox_rear, tuple)
        self.assertEqual(len(out.bbox_rear), 4)
        self.assertTrue(all(isinstance(v, float) for v in out.bbox_rear))
        self.assertIsNone(out.bbox_side)
        self.assertIsInstance(out.weight, float)
        self.assertIsNone(out.bbox)

        self.model.eval()
        self.model.predict(self.rear)
        self.assertFalse(self.model.training)


class TestDualViewSideRear(unittest.TestCase):
    """InputMode.SIDE_REAR: side + rear -> per-view bboxes + fused weight."""

    def setUp(self):
        self.side = torch.randn(1, 3, 224, 224)
        self.rear = torch.randn(1, 3, 224, 224)

    def _make_model(self, share_backbone=True):
        return DualViewTorchModel(
            architecture="mobilenet",
            variant="small",
            share_backbone=share_backbone,
            input_mode=InputMode.SIDE_REAR,
        )

    def test_two_input_forward_output_contract(self):
        model = self._make_model()
        out = model(self.side, self.rear)
        self.assertIsInstance(out, ModelOutput)
        self.assertIsInstance(out.bbox_side, torch.Tensor)
        self.assertIsInstance(out.bbox_rear, torch.Tensor)
        self.assertEqual(tuple(out.bbox_side.shape), (1, 4))
        self.assertEqual(tuple(out.bbox_rear.shape), (1, 4))
        self.assertIsInstance(out.weight, torch.Tensor)
        self.assertEqual(tuple(out.weight.shape), (1, 1))
        self.assertIsInstance(out.sex, torch.Tensor)
        self.assertEqual(tuple(out.sex.shape), (1, 2))
        self.assertIsNone(out.bbox)

    def test_batch_forward_shapes(self):
        model = self._make_model()
        out = model(torch.randn(2, 3, 224, 224), torch.randn(2, 3, 224, 224))
        self.assertEqual(tuple(out.bbox_side.shape), (2, 4))
        self.assertEqual(tuple(out.bbox_rear.shape), (2, 4))
        self.assertEqual(tuple(out.weight.shape), (2, 1))
        self.assertEqual(tuple(out.sex.shape), (2, 2))

    def test_fusion_input_dimension_matches_dual_view(self):
        model = self._make_model()
        self.assertEqual(model.fusion[0].in_features, model._backbone_feat_dim * 2)

    def test_bboxes_use_per_view_features_not_fused(self):
        model = self._make_model()
        self.assertEqual(model.bbox_side_head.in_features, model._backbone_feat_dim)
        self.assertEqual(model.bbox_rear_head.in_features, model._backbone_feat_dim)
        # The fused representation must not feed the bbox heads.
        self.assertNotEqual(model.fusion[-2].out_features, model._backbone_feat_dim)

    def test_bboxes_require_grad_and_backward(self):
        model = self._make_model()
        model.train()
        side = self.side.clone().requires_grad_(True)
        rear = self.rear.clone().requires_grad_(True)
        out = model(side, rear)
        self.assertTrue(out.bbox_side.requires_grad)
        self.assertTrue(out.bbox_rear.requires_grad)
        self.assertTrue(out.weight.requires_grad)

        loss = out.bbox_side.sum() + out.bbox_rear.sum() + out.weight.sum()
        loss.backward()
        self.assertIsNotNone(side.grad)
        self.assertIsNotNone(rear.grad)
        self.assertIsNotNone(model.bbox_side_head.weight.grad)
        self.assertIsNotNone(model.bbox_rear_head.weight.grad)
        self.assertIsNotNone(model.fusion[0].weight.grad)
        self.assertIsNotNone(model.weight_head.weight.grad)
        self.assertTrue(torch.isfinite(model.bbox_side_head.weight.grad).all())
        self.assertTrue(torch.isfinite(model.bbox_rear_head.weight.grad).all())

    def test_share_backbone_true_uses_same_module(self):
        model = self._make_model(share_backbone=True)
        self.assertIs(model.backbone_side, model.backbone_rear)

    def test_share_backbone_false_uses_independent_modules(self):
        model = self._make_model(share_backbone=False)
        self.assertIsNot(model.backbone_side, model.backbone_rear)

    def test_share_backbone_false_learns_independent_rear_weights(self):
        model = self._make_model(share_backbone=False)
        model.train()
        side = torch.randn(2, 3, 224, 224)
        rear = torch.randn(2, 3, 224, 224)
        model.zero_grad()
        model(side, rear).bbox_rear.sum().backward()
        self.assertIsNotNone(model.backbone_rear[0][0].weight.grad)
        self.assertIsNone(model.backbone_side[0][0].weight.grad)

    def test_predict_returns_python_values_and_restores_mode(self):
        model = self._make_model()
        model.train()
        out = model.predict(self.side, self.rear)
        self.assertTrue(model.training)
        for bbox in (out.bbox_side, out.bbox_rear):
            self.assertIsInstance(bbox, tuple)
            self.assertEqual(len(bbox), 4)
            self.assertTrue(all(isinstance(v, float) for v in bbox))
        self.assertIsInstance(out.weight, float)
        self.assertIsNone(out.bbox)

        model.eval()
        model.predict(self.side, self.rear)
        self.assertFalse(model.training)


class TestBackboneConstructionByInputMode(unittest.TestCase):
    """Backbone instantiation is mode-dependent: no unused view backbones."""

    def _count_module_params(self, module) -> int:
        return sum(p.numel() for p in module.parameters())

    def test_side_mode_builds_only_side_backbone(self):
        model = DualViewTorchModel(
            architecture="mobilenet", variant="small",
            share_backbone=True, input_mode=InputMode.SIDE,
        )
        self.assertIsNotNone(model.backbone_side)
        self.assertIsNone(model.backbone_rear)
        self.assertEqual(
            model.count_parameters(),
            self._count_module_params(model.backbone_side)
            + self._count_module_params(model.fusion)
            + self._count_module_params(model.bbox_side_head)
            + self._count_module_params(model.bbox_rear_head)
            + self._count_module_params(model.sex_head)
            + self._count_module_params(model.weight_head),
        )

    def test_side_mode_share_backbone_has_no_effect(self):
        for share_backbone in (True, False):
            with self.subTest(share_backbone=share_backbone):
                model = DualViewTorchModel(
                    architecture="mobilenet", variant="small",
                    share_backbone=share_backbone, input_mode=InputMode.SIDE,
                )
                self.assertIsNotNone(model.backbone_side)
                self.assertIsNone(model.backbone_rear)

    def test_rear_mode_builds_only_rear_backbone(self):
        model = DualViewTorchModel(
            architecture="mobilenet", variant="small",
            share_backbone=True, input_mode=InputMode.REAR,
        )
        self.assertIsNotNone(model.backbone_rear)
        self.assertIsNone(model.backbone_side)
        self.assertEqual(
            model.count_parameters(),
            self._count_module_params(model.backbone_rear)
            + self._count_module_params(model.fusion)
            + self._count_module_params(model.bbox_side_head)
            + self._count_module_params(model.bbox_rear_head)
            + self._count_module_params(model.sex_head)
            + self._count_module_params(model.weight_head),
        )

    def test_rear_mode_share_backbone_has_no_effect(self):
        for share_backbone in (True, False):
            with self.subTest(share_backbone=share_backbone):
                model = DualViewTorchModel(
                    architecture="mobilenet", variant="small",
                    share_backbone=share_backbone, input_mode=InputMode.REAR,
                )
                self.assertIsNotNone(model.backbone_rear)
                self.assertIsNone(model.backbone_side)

    def test_single_view_models_have_fewer_parameters_than_dual_view(self):
        side_model = DualViewTorchModel(
            architecture="mobilenet", variant="small", input_mode=InputMode.SIDE,
        )
        dual_model = DualViewTorchModel(
            architecture="mobilenet", variant="small",
            share_backbone=False, input_mode=InputMode.SIDE_REAR,
        )
        self.assertLess(side_model.count_parameters(), dual_model.count_parameters())
        self.assertLess(side_model.model_size(), dual_model.model_size())

    def test_side_rear_mode_builds_both_backbones(self):
        shared = DualViewTorchModel(
            architecture="mobilenet", variant="small",
            share_backbone=True, input_mode=InputMode.SIDE_REAR,
        )
        self.assertIsNotNone(shared.backbone_side)
        self.assertIs(shared.backbone_side, shared.backbone_rear)

        independent = DualViewTorchModel(
            architecture="mobilenet", variant="small",
            share_backbone=False, input_mode=InputMode.SIDE_REAR,
        )
        self.assertIsNotNone(independent.backbone_side)
        self.assertIsNotNone(independent.backbone_rear)
        self.assertIsNot(independent.backbone_side, independent.backbone_rear)


if __name__ == "__main__":
    unittest.main()
