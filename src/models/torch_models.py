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
  EfficientNet-B0).
- The YOLO path uses the official Ultralytics YOLO26 backbone through the
  installed `ultralytics` package; the detection head is intentionally not
  retained because this model is used as a feature extractor for the dual-view
  fusion tower.
- count_parameters() and model_size() compute real values from PyTorch
  parameters.
"""

from pathlib import Path
from typing import Any

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
            # Official Ultralytics YOLO26 exposes the detection model as a standard
            # `DetectionModel` with a backbone + neck + head pipeline. For this project,
            # we use only the inflicted backbone blocks 0..10 inclusive as a feature
            # extractor. The neck/head blocks 11..23 are intentionally omitted because
            # they are built for multi-scale detection outputs and do not act as a
            # generic image encoder for the fused dual-view classification/regression
            # tower used downstream.
            try:
                from ultralytics import YOLO as _YOLOLoader  # type: ignore
            except Exception as exc:  # pragma: no cover - import-level guard
                raise RuntimeError(
                    "Ultralytics YOLO is required for the YOLO architecture; no custom YOLO-like fallback is used."
                ) from exc

            variant_map = {
                "nano": "yolo26n.yaml",
                "small": "yolo26s.yaml",
                "medium": "yolo26m.yaml",
            }
            if variant not in variant_map:
                raise ValueError(
                    f"Unsupported YOLO variant '{variant}'. Supported variants: nano, small, medium."
                )
            config_name = variant_map[variant]

            def _make_yolo26_backbone() -> nn.Module:
                # The official YOLO26 backbone is the backbone stage defined in the
                # shipped YAML: Conv -> Conv -> C3k2 -> Conv -> C3k2 -> Conv -> C3k2
                # -> Conv -> C3k2 -> SPPF -> C2PSA. This is exactly modules 0..10
                # inclusive; modules 11+ are the neck/head stack for object detection.
                yolo = _YOLOLoader(config_name)
                backbone = nn.Sequential(*list(yolo.model.model.children())[:11])
                backbone.eval()
                return backbone

            def _infer_out_channels(mod: nn.Module) -> int:
                mod_cpu = mod.to("cpu")
                mod_cpu.eval()
                with torch.no_grad():
                    x = torch.zeros(1, 3, 224, 224)
                    out = mod_cpu(x)
                if out.ndim >= 2:
                    return int(out.shape[1])
                raise RuntimeError("YOLO26 backbone did not produce a valid feature tensor")

            self.backbone_side = _make_yolo26_backbone()
            self._backbone_feat_dim = _infer_out_channels(self.backbone_side)
            if share_backbone:
                self.backbone_rear = self.backbone_side
            else:
                self.backbone_rear = _make_yolo26_backbone()
            self.use_torchvision_backbone = False

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # The backbone for each view produces one pooled feature vector per image.
        # We concatenate the side and rear vectors to form a shared representation
        # that the learned fusion MLP can mix across both viewpoints before the task
        # heads predict bbox, sex, and weight.
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
            feats = backbone(x)
            if feats.ndim > 2:
                feats = _global_pool_flat(feats)
            return feats

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
