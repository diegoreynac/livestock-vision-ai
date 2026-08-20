from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class DetectionMetrics:
    """Container for object detection metrics.

    - IoU measures overlap between predicted and ground-truth bounding boxes.
    - mAP represents mean Average Precision for object detection.

    Actual calculation of these values will be implemented in a later task.
    """

    iou: float | None = None
    map: float | None = None


@dataclass
class ClassificationMetrics:
    """Metrics for the binary sex-classification task.

    Fields are typical binary-classification scores (accuracy, precision,
    recall, and F1). These are containers only; computation happens elsewhere.
    """

    accuracy: float | None = None
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None


@dataclass
class RegressionMetrics:
    """Containers for regression-style metrics on weight prediction.

    - MAE reports average absolute weight prediction error in kg.
    - RMSE penalizes larger errors more strongly.
    - R² measures explained variance relative to the target baseline.
    """

    mae: float | None = None
    rmse: float | None = None
    r2: float | None = None


@dataclass
class ModelComplexity:
    """Model complexity indicators.

    - total_parameters: total number of model parameters (trainable + frozen).
    - trainable_parameters: number of parameters that will be updated during
      training.
    - model_size_mb: serialized model size on disk in megabytes.
    """

    total_parameters: int | None = None
    trainable_parameters: int | None = None
    model_size_mb: float | None = None


@dataclass
class InferencePerformance:
    """Measured inference performance for a specific hardware target.

    - latency_ms: average per-sample latency in milliseconds.
    - p95_latency_ms: 95th-percentile latency in milliseconds.
    - fps: frames-per-second throughput measured on the target hardware.
    - hardware: optional description of the hardware used for measurement.

    Latency values must eventually be measured experimentally on real
    hardware; this class is a result container only.
    """

    latency_ms: float | None = None
    p95_latency_ms: float | None = None
    fps: float | None = None
    hardware: str | None = None


@dataclass(slots=True)
class ModelMetrics:
    """Combined, architecture-independent container for model evaluation

    This dataclass composes detection, classification, regression,
    complexity, and performance result containers so different model
    architectures can be compared using a single interface. It intentionally
    contains no architecture-specific behavior or calculation logic.
    """

    detection: DetectionMetrics = field(default_factory=DetectionMetrics)
    classification: ClassificationMetrics = field(default_factory=ClassificationMetrics)
    regression: RegressionMetrics = field(default_factory=RegressionMetrics)
    complexity: ModelComplexity = field(default_factory=ModelComplexity)
    performance: InferencePerformance = field(default_factory=InferencePerformance)
