import pytest
import torch

from src.models.output import ModelOutput
from src.training.losses import MultiTaskLoss, MultiTaskLossConfig


def test_default_loss_combines_bbox_and_weight() -> None:
    loss_fn = MultiTaskLoss()
    outputs = ModelOutput(
        bbox_side=torch.tensor([[0.10, 0.20, 0.80, 0.90]], requires_grad=True),
        weight=torch.tensor([[170.0]], requires_grad=True),
    )
    batch = {"bbox": torch.tensor([[0.20, 0.20, 0.80, 0.80]]), "weight": torch.tensor([[160.0]])}
    loss = loss_fn(outputs, batch)
    assert loss.ndim == 0
    assert loss.item() > 0.0
    loss.backward()
    assert outputs.bbox_side.grad is not None
    assert outputs.weight.grad is not None


def test_dual_view_bbox_losses_are_averaged() -> None:
    loss_fn = MultiTaskLoss(MultiTaskLossConfig(bbox_weight=1.0, weight_weight=0.0, bbox_loss="l1"))
    outputs = ModelOutput(
        bbox_side=torch.zeros(1, 4, requires_grad=True),
        bbox_rear=torch.zeros(1, 4, requires_grad=True),
    )
    batch = {"bbox_side": torch.ones(1, 4), "bbox_rear": torch.full((1, 4), 3.0)}
    assert loss_fn(outputs, batch).item() == pytest.approx(2.0)


@pytest.mark.parametrize("loss_name", ["l1", "smooth_l1", "mse"])
def test_configurable_regression_losses(loss_name: str) -> None:
    loss_fn = MultiTaskLoss(MultiTaskLossConfig(bbox_weight=0.0, weight_weight=1.0, weight_loss=loss_name))
    outputs = ModelOutput(weight=torch.tensor([[170.0]], requires_grad=True))
    loss = loss_fn(outputs, {"weight": torch.tensor([[160.0]])})
    assert loss.ndim == 0
    assert torch.isfinite(loss)
    loss.backward()
    assert outputs.weight.grad is not None


@pytest.mark.parametrize("bbox_weight,weight_weight", [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)])
def test_rejects_invalid_loss_weights(bbox_weight: float, weight_weight: float) -> None:
    with pytest.raises(ValueError):
        MultiTaskLossConfig(bbox_weight=bbox_weight, weight_weight=weight_weight)


def test_rejects_invalid_loss_type() -> None:
    with pytest.raises(ValueError):
        MultiTaskLossConfig(bbox_loss="cross_entropy")


def test_rejects_missing_weight_target() -> None:
    with pytest.raises(ValueError, match="weight target"):
        MultiTaskLoss()(ModelOutput(bbox_side=torch.zeros(1, 4), weight=torch.zeros(1, 1)), {"bbox": torch.zeros(1, 4)})


def test_rejects_shape_mismatch() -> None:
    outputs = ModelOutput(bbox_side=torch.zeros(2, 4), weight=torch.zeros(2, 1))
    with pytest.raises(ValueError, match="shapes must match"):
        MultiTaskLoss()(outputs, {"bbox": torch.zeros(1, 4), "weight": torch.zeros(2, 1)})


def test_rejects_missing_dual_view_bbox() -> None:
    with pytest.raises(ValueError, match="BBox target"):
        MultiTaskLoss(MultiTaskLossConfig(weight_weight=0.0))(
            ModelOutput(bbox_side=torch.zeros(1, 4)), {"weight": torch.zeros(1, 1)}
        )
