from __future__ import annotations

"""PyTorch-backed model implementations for dual-view processing.

These classes provide concrete, real neural-network implementations that
integrate with the project's BaseModel / ModelOutput interfaces. They are
intentionally framework-specific (PyTorch) and meant to be used when a
PyTorch runtime is available.

Design summary
- Side and Rear images are processed independently by identical backbone
  modules (one per view). Backbones can be chosen via the `architecture`
  argument and support configurable `variant` strings.
- Feature fusion is implemented as concatenation of the per-view feature
  vectors followed by a small learned fusion MLP.
- Three small task-specific heads map the fused representation to:
  - bounding box regression (4 floats)
  - sex classification (2 logits -> 'F' or 'M')
  - weight regression (1 float)

Notes
- This module uses torchvision backbones when available (MobileNetV3 and
  EfficientNet-B0). For YOLO, a lightweight custom convolutional backbone is
  implemented to avoid adding a heavy external YOLO package.
- count_parameters() and model_size() compute real values from PyTorch
  parameters.
"""

from pathlib import Path
from typing import Any, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from torchvision import models as tv_models
except Exception:
    tv_models = None  # runtime import guard; tests will fail if torch isn't installed

from src.models.base import BaseModel
from src.models.output import ModelOutput


def _global_pool_flat(x: torch.Tensor) -> torch.Tensor:
    # Global average pool to (batch, channels)
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


class _YOLOLikeBackbone(nn.Module):
    """A small YOLO-like backbone implemented with plain convolutions.

    This backbone produces a compact feature vector suitable for fusion and
    downstream heads. It is deliberately lightweight and does not implement a
    full YOLO detection head — the detection head is implemented separately
    in the DualViewTorchModel's bbox head.
    """

    def __init__(self, widths: Tuple[int, ...] = (16, 32, 64, 128)) -> None:
        super().__init__()
        layers = []
        in_ch = 3
        for w in widths:
            layers.append(nn.Conv2d(in_ch, w, kernel_size=3, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(w))
            layers.append(nn.ReLU(inplace=True))
            in_ch = w
        self.features = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return _global_pool_flat(x)


class DualViewTorchModel(BaseModel, nn.Module):
    """Dual-view PyTorch model implementing backbone, fusion and multi-task heads.

    Parameters
    - architecture: 'mobilenet', 'efficientnet' or 'yolo'
    - variant: variant string passed to the backbone selection
    - share_backbone: if True, the side and rear views share backbone weights

    The forward method accepts two image tensors: side and rear with shape
    (B, C, H, W) and returns a ModelOutput with framework-agnostic values.
    """

    def __init__(
        self,
        architecture: str = "mobilenet",
        variant: str = "default",
        share_backbone: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self.architecture = architecture
        self.variant = variant
        self.share_backbone = share_backbone

        # instantiate backbones
        if architecture == "mobilenet":
            if tv_models is None:
                raise RuntimeError("torchvision is required for MobileNet backbones")
            # MobileNetV3 small/large mapping
            if variant in ("small", "v3-small"):
                net = tv_models.mobilenet_v3_small(pretrained=False)
                feat_dim = 576  # mobilenet_v3_small final feature channels before classifier
            else:
                net = tv_models.mobilenet_v3_large(pretrained=False)
                feat_dim = 960

            # use feature extractor portion
            self.backbone_side = net.features
            self._backbone_feat_dim = feat_dim
            if share_backbone:
                self.backbone_rear = self.backbone_side
            else:
                # create a separate copy (same architecture)
                net2 = tv_models.mobilenet_v3_small(pretrained=False) if variant in ("small", "v3-small") else tv_models.mobilenet_v3_large(pretrained=False)
                self.backbone_rear = net2.features

            # backbones produce a feature map; we will global-pool to vector
            self.use_torchvision_backbone = True

        elif architecture == "efficientnet":
            if tv_models is None:
                raise RuntimeError("torchvision is required for EfficientNet backbones")
            # map variant names to torchvision EfficientNet variants
            if variant in ("b0", "default"):
                net = tv_models.efficientnet_b0(pretrained=False)
                feat_dim = 1280
            else:
                # fallback: b0
                net = tv_models.efficientnet_b0(pretrained=False)
                feat_dim = 1280

            self.backbone_side = net.features
            self._backbone_feat_dim = feat_dim
            if share_backbone:
                self.backbone_rear = self.backbone_side
            else:
                net2 = tv_models.efficientnet_b0(pretrained=False)
                self.backbone_rear = net2.features
            self.use_torchvision_backbone = True

        elif architecture == "yolo":
            # lightweight custom YOLO-like backbone implemented above
            widths_map = {
                "nano": (16, 32, 64, 128),
                "small": (32, 64, 128, 256),
                "medium": (64, 128, 256, 512),
            }
            widths = widths_map.get(variant, (16, 32, 64, 128))
            self.backbone_side = _YOLOLikeBackbone(widths)
            self._backbone_feat_dim = widths[-1]
            if share_backbone:
                self.backbone_rear = self.backbone_side
            else:
                self.backbone_rear = _YOLOLikeBackbone(widths)
            self.use_torchvision_backbone = False

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # fusion MLP: projects concatenated features to a compact fusion dimension
        fusion_in = self._backbone_feat_dim * 2
        fusion_dim = max(128, fusion_in // 4)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # heads
        self.bbox_head = nn.Linear(fusion_dim, 4)
        self.sex_head = nn.Linear(fusion_dim, 2)
        self.weight_head = nn.Linear(fusion_dim, 1)

    def _extract_features(self, backbone: nn.Module, x: torch.Tensor) -> torch.Tensor:
        """Run the backbone and return a pooled feature vector (B, C)."""
        if self.use_torchvision_backbone:
            # torchvision backbones expect images normalized; tests supply random tensors
            feats = backbone(x)  # (B, C, H, W)
            vec = _global_pool_flat(feats)
            return vec
        else:
            return backbone(x)

    def forward(self, side: Any, rear: Any, **kwargs: Any) -> ModelOutput:
        """Perform a PyTorch forward pass.

        Inputs are expected to be torch.Tensor with shape (B, C, H, W) or
        convertible to such. The implementation runs on CPU by default but will
        use default device of provided tensors.
        """
        # Accept raw tensors or convert numpy-like arrays here if necessary
        if not isinstance(side, torch.Tensor) or not isinstance(rear, torch.Tensor):
            raise TypeError("side and rear inputs must be torch.Tensor for DualViewTorchModel")

        device = side.device
        side_feats = self._extract_features(self.backbone_side, side.to(device))
        rear_feats = self._extract_features(self.backbone_rear, rear.to(device))

        fused = torch.cat([side_feats, rear_feats], dim=1)
        fused = self.fusion(fused)

        bbox = self.bbox_head(fused)  # (B,4)
        sex_logits = self.sex_head(fused)  # (B,2)
        weight_out = self.weight_head(fused)  # (B,1)

        # For interface compatibility, convert single-batch tensors to plain Python
        # scalars/lists. Keep batch dimension if B>1.
        if bbox.shape[0] == 1:
            bbox_val = tuple(float(x) for x in bbox.squeeze(0).tolist())
            sex_idx = int(sex_logits.argmax(dim=1).item())
            sex_val = "M" if sex_idx == 1 else "F"
            weight_val = float(weight_out.squeeze(0).item())
        else:
            bbox_val = [tuple(float(x) for x in row.tolist()) for row in bbox]
            sex_val = ["M" if int(idx) == 1 else "F" for idx in sex_logits.argmax(dim=1).tolist()]
            weight_val = [float(x) for x in weight_out.flatten().tolist()]

        return ModelOutput(bbox=bbox_val, sex=sex_val, weight=weight_val)

    def predict(self, side: Any, rear: Any, **kwargs: Any) -> ModelOutput:
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                return self.forward(side, rear, **kwargs)
        finally:
            if was_training:
                self.train()

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())

    def model_size(self) -> float:
        # bytes = params * 4 (float32)
        params = self.count_parameters()
        mb = (params * 4) / (1024.0 * 1024.0)
        return float(mb)

    def export(self, destination: Path | str, **kwargs: Any) -> None:
        dest = Path(destination)
        dest.mkdir(parents=True, exist_ok=True)
        # Save state_dict
        torch.save(self.state_dict(), str(dest / "model_state.pth"))
        metadata = (
            f"architecture={self.architecture}\n"
            f"variant={self.variant}\n"
            f"parameters={self.count_parameters()}\n"
        )
        (dest / "model_metadata.txt").write_text(metadata, encoding="utf-8")
