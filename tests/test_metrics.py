import unittest

from src.evaluation.metrics import (
    DetectionMetrics,
    ClassificationMetrics,
    RegressionMetrics,
    ModelComplexity,
    InferencePerformance,
    ModelMetrics,
)


class TestMetrics(unittest.TestCase):
    def test_detection_metrics_defaults_and_explicit(self):
        d = DetectionMetrics()
        self.assertIsNone(d.iou)
        self.assertIsNone(d.map)

        d2 = DetectionMetrics(iou=0.5, map=0.6)
        self.assertEqual(d2.iou, 0.5)
        self.assertEqual(d2.map, 0.6)

    def test_classification_metrics_defaults_and_explicit(self):
        c = ClassificationMetrics()
        self.assertIsNone(c.accuracy)
        self.assertIsNone(c.precision)
        self.assertIsNone(c.recall)
        self.assertIsNone(c.f1)

        c2 = ClassificationMetrics(accuracy=0.8, precision=0.75, recall=0.7, f1=0.72)
        self.assertEqual(c2.accuracy, 0.8)
        self.assertEqual(c2.precision, 0.75)
        self.assertEqual(c2.recall, 0.7)
        self.assertEqual(c2.f1, 0.72)

    def test_regression_metrics_defaults_and_explicit(self):
        r = RegressionMetrics()
        self.assertIsNone(r.mae)
        self.assertIsNone(r.rmse)
        self.assertIsNone(r.r2)

        r2 = RegressionMetrics(mae=1.2, rmse=1.5, r2=0.85)
        self.assertEqual(r2.mae, 1.2)
        self.assertEqual(r2.rmse, 1.5)
        self.assertEqual(r2.r2, 0.85)

    def test_model_complexity_defaults_and_explicit(self):
        m = ModelComplexity()
        self.assertIsNone(m.total_parameters)
        self.assertIsNone(m.trainable_parameters)
        self.assertIsNone(m.model_size_mb)

        m2 = ModelComplexity(total_parameters=1000, trainable_parameters=800, model_size_mb=12.3)
        self.assertEqual(m2.total_parameters, 1000)
        self.assertEqual(m2.trainable_parameters, 800)
        self.assertEqual(m2.model_size_mb, 12.3)

    def test_inference_performance_defaults_and_explicit(self):
        p = InferencePerformance()
        self.assertIsNone(p.latency_ms)
        self.assertIsNone(p.p95_latency_ms)
        self.assertIsNone(p.fps)
        self.assertIsNone(p.hardware)

        p2 = InferencePerformance(latency_ms=10.0, p95_latency_ms=20.0, fps=30.0, hardware="cpu-x86")
        self.assertEqual(p2.latency_ms, 10.0)
        self.assertEqual(p2.p95_latency_ms, 20.0)
        self.assertEqual(p2.fps, 30.0)
        self.assertEqual(p2.hardware, "cpu-x86")

    def test_model_metrics_composition_and_defaults(self):
        mm = ModelMetrics()
        self.assertIsInstance(mm.detection, DetectionMetrics)
        self.assertIsInstance(mm.classification, ClassificationMetrics)
        self.assertIsInstance(mm.regression, RegressionMetrics)
        self.assertIsInstance(mm.complexity, ModelComplexity)
        self.assertIsInstance(mm.performance, InferencePerformance)

        # defaults inside nested objects should remain None
        self.assertIsNone(mm.detection.iou)
        self.assertIsNone(mm.classification.accuracy)
        self.assertIsNone(mm.regression.mae)

    def test_model_metrics_explicit_preserved(self):
        mm = ModelMetrics(
            detection=DetectionMetrics(iou=0.9),
            classification=ClassificationMetrics(accuracy=0.77),
            regression=RegressionMetrics(mae=0.5),
            complexity=ModelComplexity(total_parameters=123456),
            performance=InferencePerformance(latency_ms=5.0),
        )

        self.assertEqual(mm.detection.iou, 0.9)
        self.assertEqual(mm.classification.accuracy, 0.77)
        self.assertEqual(mm.regression.mae, 0.5)
        self.assertEqual(mm.complexity.total_parameters, 123456)
        self.assertEqual(mm.performance.latency_ms, 5.0)


if __name__ == "__main__":
    unittest.main()
