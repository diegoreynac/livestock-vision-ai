import pytest
from pathlib import Path

import torch
from torch import nn

from src.models.torch_models import ModelOutput
from src.training.checkpoint import CheckpointManager
from src.training.losses import MultiTaskLoss, MultiTaskLossConfig
from src.training.trainer import Trainer

from src.models.output import ModelOutput


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

class TinySingleViewModel(nn.Module):
    """Small trainable model for end-to-end single-view loss tests."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 8),
            nn.ReLU(),
        )
        self.bbox_head = nn.Linear(8, 4)
        self.weight_head = nn.Linear(8, 1)

    def forward(self, image: torch.Tensor) -> ModelOutput:
        features = self.features(image)
        return ModelOutput(
            bbox_side=self.bbox_head(features),
            weight=self.weight_head(features),
        )


class TinyDualViewModel(nn.Module):
    """Small trainable model for end-to-end dual-view loss tests."""

    def __init__(self) -> None:
        super().__init__()
        self.side_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 8),
            nn.ReLU(),
        )
        self.rear_features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 8 * 8, 8),
            nn.ReLU(),
        )
        self.side_bbox_head = nn.Linear(8, 4)
        self.rear_bbox_head = nn.Linear(8, 4)
        self.weight_head = nn.Linear(16, 1)

    def forward(
        self,
        side: torch.Tensor,
        rear: torch.Tensor,
    ) -> ModelOutput:
        side_features = self.side_features(side)
        rear_features = self.rear_features(rear)

        fused_features = torch.cat(
            [side_features, rear_features],
            dim=1,
        )

        return ModelOutput(
            bbox_side=self.side_bbox_head(side_features),
            bbox_rear=self.rear_bbox_head(rear_features),
            weight=self.weight_head(fused_features),
        )

class DummyLivestockDataset(torch.utils.data.Dataset):
    def __init__(self, images, bbox, weight):
        self.images = images
        self.bbox = bbox
        self.weight = weight

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        return {
            "image": self.images[index],
            "bbox": self.bbox[index],
            "weight": self.weight[index],
        }

def _single_view_batch() -> dict[str, object]:
    return {
        "image": torch.full((2, 3, 8, 8), 0.25),
        "bbox": torch.tensor(
            [
                [0.20, 0.20, 0.60, 0.60],
                [0.25, 0.15, 0.55, 0.70],
            ],
            dtype=torch.float32,
        ),
        "weight": torch.tensor(
            [[1.0], [1.5]],
            dtype=torch.float32,
        ),
        "animal_id": ["animal-001", "animal-002"],
    }


def _dual_view_batch() -> dict[str, object]:
    return {
        "side_image": torch.full((2, 3, 8, 8), 0.25),
        "rear_image": torch.full((2, 3, 8, 8), 0.75),
        "bbox_side": torch.tensor(
            [
                [0.20, 0.20, 0.60, 0.60],
                [0.25, 0.15, 0.55, 0.70],
            ],
            dtype=torch.float32,
        ),
        "bbox_rear": torch.tensor(
            [
                [0.15, 0.25, 0.70, 0.55],
                [0.20, 0.20, 0.65, 0.60],
            ],
            dtype=torch.float32,
        ),
        "weight": torch.tensor(
            [[1.0], [1.5]],
            dtype=torch.float32,
        ),
        "animal_id": ["animal-001", "animal-002"],
    }

def test_multitask_loss_end_to_end_single_view_updates_model(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model = TinySingleViewModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=MultiTaskLoss(),
        checkpoint_manager=CheckpointManager(tmp_path),
    )

    batch = _single_view_batch()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = trainer.train_epoch([batch])

    assert result.loss >= 0.0
    assert torch.isfinite(torch.tensor(result.loss))

    changed_parameters = [
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    ]

    assert changed_parameters

def test_multitask_loss_end_to_end_dual_view_updates_all_heads(
    tmp_path: Path,
) -> None:
    torch.manual_seed(42)

    model = TinyDualViewModel()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-2,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=MultiTaskLoss(),
        checkpoint_manager=CheckpointManager(tmp_path),
    )

    batch = _dual_view_batch()

    before = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    result = trainer.train_epoch([batch])

    assert result.loss >= 0.0
    assert torch.isfinite(torch.tensor(result.loss))

    changed_parameters = {
        name
        for name, parameter in model.named_parameters()
        if not torch.equal(before[name], parameter.detach())
    }

    assert changed_parameters

    assert any(
        name.startswith("side_bbox_head")
        for name in changed_parameters
    )
    assert any(
        name.startswith("rear_bbox_head")
        for name in changed_parameters
    )
    assert any(
        name.startswith("weight_head")
        for name in changed_parameters
    )

def test_multitask_loss_dual_view_uses_both_bbox_targets() -> None:
    torch.manual_seed(42)

    model = TinyDualViewModel()
    loss_fn = MultiTaskLoss()

    side_image = torch.full((2, 3, 8, 8), 0.25)
    rear_image = torch.full((2, 3, 8, 8), 0.75)

    outputs = model(side_image, rear_image)

    batch = _dual_view_batch()

    original_loss = loss_fn(outputs, batch)

    modified_batch = {
        **batch,
        "bbox_rear": torch.zeros_like(batch["bbox_rear"]),
    }

    modified_loss = loss_fn(outputs, modified_batch)

    assert not torch.equal(original_loss, modified_loss)

def test_mobilenet_dummy_training():
    """Verify that MobileNetV3 Small can complete a short training run."""
    import torch
    from torch.utils.data import DataLoader

    from src.models.torch_models import DualViewTorchModel, InputMode
    from src.training.config import TrainingConfig
    from src.training.losses import MultiTaskLoss, MultiTaskLossConfig
    from src.training.trainer import Trainer
    from src.training.optimizer import OptimizerFactory

    torch.manual_seed(42)

    batch_size = 2
    num_samples = 4

    images = torch.randn(num_samples, 3, 224, 224)
    bbox = torch.tensor(
        [
            [0.10, 0.10, 0.80, 0.80],
            [0.15, 0.15, 0.75, 0.75],
            [0.20, 0.20, 0.70, 0.70],
            [0.25, 0.25, 0.65, 0.65],
        ],
        dtype=torch.float32,
    )
    weight = torch.tensor(
        [[120.0], [140.0], [160.0], [180.0]],
        dtype=torch.float32,
    )

    dataset = DummyLivestockDataset(images, bbox, weight)

    model = DualViewTorchModel(
        architecture="mobilenet",
        variant="small",
        input_mode=InputMode.SIDE,
        pretrained=True,
    )

    loss_fn = MultiTaskLoss(
        MultiTaskLossConfig(
            bbox_weight=1.0,
            weight_weight=1.0,
        )
    )

    config = TrainingConfig(
        seed=42,
        input_size=(224, 224),
        batch_size=batch_size,
        epochs=2,
        learning_rate=1e-3,
        weight_decay=1e-4,
        device="cpu",
    )

    optimizer = OptimizerFactory.create(model, config)

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        device=config.device,
    )   

    initial_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    result = trainer.fit(
        train_dataloader=dataloader,
        validation_dataloader=dataloader,
        epochs=config.epochs,
    )

    assert result is not None

    final_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }

    parameters_changed = any(
        not torch.equal(initial_parameters[name], final_parameters[name])
        for name in initial_parameters
    )

    assert parameters_changed