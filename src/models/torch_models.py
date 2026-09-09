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
- The active set of views is selected with `input_mode` (InputMode.SIDE,
  InputMode.REAR or InputMode.SIDE_REAR). Single-view modes run one backbone;
  the dual-view mode runs both.
- BBox regression is per-view: each view's own pooled feature vector feeds a
  per-view bbox head, because side and rear images live in different
  coordinate systems.
- Weight regression uses a fusion MLP whose input dimension matches the mode:
  the single pooled feature vector for single-view modes, or the
  concatenation of both view vectors for the dual-view mode. Single-view
  features are never duplicated to simulate a dual-view input.
- A small sex head on the fused representation is retained temporarily for
  API compatibility only; no sex loss or training logic exists.

Notes
- This module uses torchvision backbones when available (MobileNetV3 and
  EfficientNet-B0).
- The YOLO path uses the official Ultralytics YOLO26 backbone through the
  installed `ultralytics` package; the detection head is intentionally not
  retained because this model is used as a feature extractor for the
  regression tower.
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
from src.training.torch_dataset import InputMode


def _global_pool_flat(x: torch.Tensor) -> torch.Tensor:
    # Global average pool to (batch, channels)
    return F.adaptive_avg_pool2d(x, 1).flatten(1)


class DualViewTorchModel(BaseModel, nn.Module):
    """Dual-view PyTorch model implementing backbone, fusion and multi-task heads.

    Parameters
    - architecture: 'mobilenet', 'efficientnet' or 'yolo'
    - variant: variant string passed to the backbone selection
    - share_backbone: if True, the side and rear views share backbone weights
    - input_mode: InputMode selecting the forward()/predict() input contract.
      SIDE_REAR (default) processes both views; SIDE and REAR run the
      corresponding single-view path.

    The forward method accepts image tensors with shape (B, C, H, W) matching
    the configured input mode and returns a ModelOutput with
    framework-agnostic values.
    """

    def __init__(
        self,
        architecture: str = "mobilenet",
        variant: str = "default",
        share_backbone: bool = False,
        input_mode: InputMode = InputMode.SIDE_REAR,
    ) -> None:
        nn.Module.__init__(self)
        if not isinstance(input_mode, InputMode):
            raise TypeError(
                "input_mode must be an InputMode instance; "
                f"got {type(input_mode).__name__}: {input_mode!r}."
            )
        self.architecture = architecture
        self.variant = variant
        self.share_backbone = share_backbone
        self.input_mode = input_mode

        # Build per-view backbones on demand. Only the views required by the
        # input mode are constructed so that count_parameters()/model_size()
        # reflect the architecture actually used; the unused view's attribute
        # is set to None (no dummy/unused modules are kept around).
        if architecture == "mobilenet":
            if tv_models is None:
                raise RuntimeError("torchvision is required for MobileNet backbones")
            # MobileNetV3 small/large mapping
            if variant in ("small", "v3-small"):
                factory = tv_models.mobilenet_v3_small
                feat_dim = 576  # mobilenet_v3_small final feature channels before classifier
            else:
                factory = tv_models.mobilenet_v3_large
                feat_dim = 960

            def _make_backbone() -> nn.Module:
                # use feature extractor portion; we global-pool the feature map
                return factory(pretrained=False).features

            self.use_torchvision_backbone = True

        elif architecture == "efficientnet":
            if tv_models is None:
                raise RuntimeError("torchvision is required for EfficientNet backbones")
            # map variant names to torchvision EfficientNet variants
            factory = tv_models.efficientnet_b0  # b0 is the fallback for any variant
            feat_dim = 1280

            def _make_backbone() -> nn.Module:
                return factory(pretrained=False).features

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

            _make_backbone = _make_yolo26_backbone
            feat_dim = _infer_out_channels(_make_yolo26_backbone())
            self.use_torchvision_backbone = False

        else:
            raise ValueError(f"Unknown architecture: {architecture}")

        # Construct only the backbones the input mode needs. share_backbone is
        # a dual-view-only concern; single-view modes use exactly one backbone.
        if input_mode is InputMode.SIDE:
            self.backbone_side = _make_backbone()
            self.backbone_rear = None
        elif input_mode is InputMode.REAR:
            self.backbone_side = None
            self.backbone_rear = _make_backbone()
        else:
            self.backbone_side = _make_backbone()
            self.backbone_rear = self.backbone_side if share_backbone else _make_backbone()
        self._backbone_feat_dim = feat_dim

        # The backbone for each view produces one pooled feature vector per
        # image. BBox regression is per-view: each head consumes only its own
        # view's features because the side and rear images live in different
        # coordinate systems. The weight/sex heads consume the fused
        # representation, whose input dimension matches the input mode: one
        # feature vector for single-view modes, the concatenation of both
        # view vectors for the dual-view mode.
        fusion_in = (
            self._backbone_feat_dim
            if input_mode is not InputMode.SIDE_REAR
            else self._backbone_feat_dim * 2
        )
        fusion_dim = max(128, fusion_in // 4)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_in, fusion_dim),
            nn.ReLU(inplace=True),
            nn.Linear(fusion_dim, fusion_dim),
            nn.ReLU(inplace=True),
        )

        # heads
        self.bbox_side_head = nn.Linear(self._backbone_feat_dim, 4)
        self.bbox_rear_head = nn.Linear(self._backbone_feat_dim, 4)
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

    def forward(self, side: Any = None, rear: Any = None, **kwargs: Any) -> ModelOutput:
        """Perform a differentiable PyTorch forward pass.

        The number of image tensors must match ``self.input_mode`` exactly:
        - InputMode.SIDE: exactly one tensor, the side view.
        - InputMode.REAR: exactly one tensor, the rear view.
        - InputMode.SIDE_REAR: exactly two tensors, side and rear.

        A TypeError is raised when the input contract is violated; inputs are
        never silently ignored and a missing view is never substituted.

        Inputs are expected to be torch.Tensor with shape (B, C, H, W). The
        implementation runs on CPU by default but will use the default device
        of the provided tensors.

        Returns a ModelOutput whose fields are the raw head tensors so the
        autograd graph is preserved for gradient-based training:
        - bbox_side: Tensor of shape (B, 4), or None in REAR mode
        - bbox_rear: Tensor of shape (B, 4), or None in SIDE mode
        - weight: Tensor of shape (B, 1)
        - sex: Tensor of shape (B, 2) with sex logits (API compatibility only)

        Use predict() for inference-friendly Python values.
        """
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected forward() keyword argument(s): {unexpected}.")

        if self.input_mode is InputMode.SIDE_REAR:
            if side is None or rear is None:
                raise TypeError(
                    "InputMode.SIDE_REAR requires exactly two tensors: forward(side, rear)."
                )
            if not isinstance(side, torch.Tensor) or not isinstance(rear, torch.Tensor):
                raise TypeError("side and rear inputs must be torch.Tensor for DualViewTorchModel")
            return self._forward_dual_view(side, rear)

        # Single-view modes consume exactly one image tensor.
        views = [view for view in (side, rear) if view is not None]
        if len(views) != 1:
            raise TypeError(
                f"InputMode.{self.input_mode.name} requires exactly one tensor; "
                f"got {len(views)}."
            )
        if not isinstance(views[0], torch.Tensor):
            raise TypeError(
                f"InputMode.{self.input_mode.name} input must be a torch.Tensor; "
                f"got {type(views[0]).__name__}."
            )
        return self._forward_single_view(views[0])

    def _forward_single_view(self, view: torch.Tensor) -> ModelOutput:
        """Single-view forward path (InputMode.SIDE / InputMode.REAR).

        side -> side_backbone -> side_features ┬-> bbox_side_head -> bbox_side
                                               └-> fusion -> weight/sex heads
        rear -> rear_backbone -> rear_features ┬-> bbox_rear_head -> bbox_rear
                                               └-> fusion -> weight/sex heads
        """
        if self.input_mode is InputMode.SIDE:
            feats = self._extract_features(self.backbone_side, view)
            bbox_side = self.bbox_side_head(feats)  # (B,4)
            bbox_rear = None
        else:
            feats = self._extract_features(self.backbone_rear, view)
            bbox_side = None
            bbox_rear = self.bbox_rear_head(feats)  # (B,4)

        fused = self.fusion(feats)
        sex_logits = self.sex_head(fused)  # (B,2)
        weight_out = self.weight_head(fused)  # (B,1)

        return ModelOutput(bbox_side=bbox_side, bbox_rear=bbox_rear, weight=weight_out, sex=sex_logits)

    def _forward_dual_view(self, side: torch.Tensor, rear: torch.Tensor) -> ModelOutput:
        """Dual-view forward path (InputMode.SIDE_REAR).

        side -> side_backbone -> side_features -> bbox_side_head -> bbox_side
                                          \
                                           concat -> fusion -> weight/sex heads
                                          /
        rear -> rear_backbone -> rear_features -> bbox_rear_head -> bbox_rear
        """
        device = side.device
        side_feats = self._extract_features(self.backbone_side, side.to(device))
        rear_feats = self._extract_features(self.backbone_rear, rear.to(device))

        bbox_side = self.bbox_side_head(side_feats)  # (B,4)
        bbox_rear = self.bbox_rear_head(rear_feats)  # (B,4)

        fused = torch.cat([side_feats, rear_feats], dim=1)
        fused = self.fusion(fused)

        sex_logits = self.sex_head(fused)  # (B,2)
        weight_out = self.weight_head(fused)  # (B,1)

        return ModelOutput(bbox_side=bbox_side, bbox_rear=bbox_rear, weight=weight_out, sex=sex_logits)

    @staticmethod
    def _bbox_to_python(bbox: torch.Tensor | None) -> Any:
        """Convert a (B, 4) bbox tensor to Python values; None stays None."""
        if bbox is None:
            return None
        if bbox.shape[0] == 1:
            return tuple(float(x) for x in bbox.squeeze(0).tolist())
        return [tuple(float(x) for x in row.tolist()) for row in bbox]

    def predict(self, side: Any = None, rear: Any = None, **kwargs: Any) -> ModelOutput:
        was_training = self.training
        try:
            self.eval()
            with torch.no_grad():
                out = self.forward(side, rear, **kwargs)
        finally:
            if was_training:
                self.train()

        weight_out = out.weight

        # Convert tensors to plain Python values for the inference interface.
        # Keep single-sample outputs unbatched; use lists when B>1.
        if weight_out.shape[0] == 1:
            weight_val: Any = float(weight_out.squeeze(0).item())
        else:
            weight_val = [float(x) for x in weight_out.flatten().tolist()]

        return ModelOutput(
            bbox_side=self._bbox_to_python(out.bbox_side),
            bbox_rear=self._bbox_to_python(out.bbox_rear),
            weight=weight_val,
            sex=out.sex,
        )

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
