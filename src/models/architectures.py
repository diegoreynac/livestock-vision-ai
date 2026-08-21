from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class ArchitectureSpec:
    """Lightweight, framework-agnostic spec for a model architecture variant.

    This object encodes a sequence of "layers" represented as (in_features,
    out_features) tuples. It provides parameter counting and a simple model-size
    estimate in megabytes assuming 4-byte floats and one bias per output unit.

    The intention is to be independent of any particular deep-learning
    framework: these specs are used by higher-level model wrappers to expose
    counts, sizes and configurable variants without importing torch/TF.
    """

    name: str
    variant: str
    layer_shapes: List[Tuple[int, int]]

    def count_parameters(self) -> int:
        """Compute total parameter count from layer shapes.

        Parameters are approximated as (in_features * out_features) + out_features
        to account for biases. This is deterministic and reproducible across
        environments.
        """
        total = 0
        for in_f, out_f in self.layer_shapes:
            total += in_f * out_f + out_f
        return total

    def model_size_mb(self) -> float:
        """Estimate model size in megabytes assuming 4 bytes per parameter."""
        params = self.count_parameters()
        bytes_ = params * 4
        mb = bytes_ / (1024.0 * 1024.0)
        return mb


# Factory helpers -----------------------------------------------------------

def _make_from_widths(name: str, variant: str, widths: List[int]) -> ArchitectureSpec:
    """Utility: build a simple sequential architecture spec from a list of
    channel widths. Each adjacent pair defines a layer (in, out)."""

    shapes: List[Tuple[int, int]] = []
    for i in range(len(widths) - 1):
        shapes.append((widths[i], widths[i + 1]))
    return ArchitectureSpec(name=name, variant=variant, layer_shapes=shapes)


def mobilenet(variant: str = "v1") -> ArchitectureSpec:
    """Create a MobileNet-like spec.

    Variants are lightweight presets mapping to interior channel-width lists.
    These are intentionally simplified and not intended to be exact reproductions
    of the original MobileNet family — the goal is deterministic parameter
    counting and size estimation for tests and tooling.
    """
    if variant in ("v1", "default"):
        widths = [32, 64, 128, 128, 256, 256, 512]
    elif variant == "small":
        widths = [16, 32, 64, 64, 128]
    elif variant == "large":
        widths = [64, 128, 256, 256, 512, 512, 1024]
    else:
        raise ValueError(f"Unknown MobileNet variant: {variant}")
    return _make_from_widths("MobileNet", variant, widths)


def efficientnet(variant: str = "b0") -> ArchitectureSpec:
    """Create an EfficientNet-like spec.

    The variants here are coarse-grained presets to allow configuration in
    tests. They are not tied to any external dependency.
    """
    if variant in ("b0", "default"):
        widths = [32, 16, 24, 40, 80, 112, 192, 320]
    elif variant == "small":
        widths = [16, 8, 16, 24, 40, 56, 96]
    elif variant == "large":
        widths = [64, 32, 64, 128, 256, 352, 640]
    else:
        raise ValueError(f"Unknown EfficientNet variant: {variant}")
    return _make_from_widths("EfficientNet", variant, widths)


def yolo(variant: str = "nano") -> ArchitectureSpec:
    """Create a YOLO-like spec.

    These presets capture the intuition of a small-to-medium detection backbone.
    They purposely include a few downsampling/upsampling stages by varying
    channel widths. Again, these are simplified and framework-agnostic.
    """
    if variant in ("nano", "default"):
        widths = [16, 32, 64, 128]
    elif variant == "small":
        widths = [32, 64, 128, 256]
    elif variant == "medium":
        widths = [64, 128, 256, 512]
    else:
        raise ValueError(f"Unknown YOLO variant: {variant}")
    return _make_from_widths("YOLO", variant, widths)
