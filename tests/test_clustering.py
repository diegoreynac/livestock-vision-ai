import unittest

from src.analysis.clustering import (
    ClusteringAnalyzer,
    ClusteringConfig,
    ClusteringMetrics,
    ClusteringReport,
    ClusterResult,
    FeatureVector,
)


class TestClusteringModule(unittest.TestCase):
    def test_clustering_config_validation(self) -> None:
        config = ClusteringConfig(
            analysis_level="image",
            reduction_method="none",
            clustering_method="none",
            n_clusters=None,
            random_seed=123,
        )

        self.assertEqual(config.analysis_level, "image")
        self.assertEqual(config.clustering_method, "none")
        self.assertEqual(config.random_seed, 123)

    def test_feature_vector_validation(self) -> None:
        vector = FeatureVector(
            sample_id="img1",
            embedding=[0.1, 0.2, 0.3],
            filename="img1.jpg",
            animal_id="A",
            metadata={"view": "Side"},
        )

        vector.validate()
        self.assertEqual(vector.to_dict()["sample_id"], "img1")

    def test_feature_vector_invalid_embedding_raises(self) -> None:
        vector = FeatureVector(sample_id="img1", embedding=[0.1, float("nan")])

        with self.assertRaises(ValueError):
            vector.validate()

    def test_clustering_analyzer_with_dummy_clusterer(self) -> None:
        config = ClusteringConfig(clustering_method="none")
        analyzer = ClusteringAnalyzer(config)

        vectors = [
            FeatureVector(sample_id="img1", embedding=[0.0, 0.0]),
            FeatureVector(sample_id="img2", embedding=[1.0, 1.0]),
        ]

        report = analyzer.analyze(vectors)

        self.assertIsInstance(report, ClusteringReport)
        self.assertEqual(report.clustering_method, "none")
        self.assertEqual(report.metrics.n_clusters, 1)
        self.assertEqual(report.metrics.n_samples, 2)
        self.assertEqual(report.metrics.n_outliers, 0)
        self.assertEqual(len(report.results), 2)
        self.assertEqual(report.results[0].cluster_id, 0)

    def test_clustering_report_contains_excluded_variables(self) -> None:
        config = ClusteringConfig(
            clustering_method="none",
            supervised_exclusions=["weight_kg", "sex"],
        )
        analyzer = ClusteringAnalyzer(config)

        vectors = [FeatureVector(sample_id="img1", embedding=[0.5])]
        report = analyzer.analyze(vectors)

        self.assertEqual(report.excluded_variables, ["weight_kg", "sex"])

    def test_clustering_metrics_none_for_no_clustering(self) -> None:
        config = ClusteringConfig(clustering_method="none", n_clusters=5)
        analyzer = ClusteringAnalyzer(config)

        vectors = [
            FeatureVector(sample_id="img1", embedding=[0.0, 0.0]),
            FeatureVector(sample_id="img2", embedding=[1.0, 1.0]),
        ]

        report = analyzer.analyze(vectors)

        self.assertIsNone(report.metrics.silhouette_score)
        self.assertIsNone(report.metrics.calinski_harabasz_score)
        self.assertIsNone(report.metrics.davies_bouldin_score)
        self.assertEqual(report.metrics.n_clusters, 1)
        self.assertEqual(report.metrics.n_outliers, 0)
        self.assertEqual(report.metrics.n_samples, 2)
        self.assertIn("no algorithmic clustering has been executed", report.metadata["note"])

    def test_clustering_method_not_implemented_raises(self) -> None:
        config = ClusteringConfig(clustering_method="kmeans", n_clusters=2)
        analyzer = ClusteringAnalyzer(config)
        vectors = [FeatureVector(sample_id="img1", embedding=[0.0, 0.0])]

        with self.assertRaises(NotImplementedError):
            analyzer.analyze(vectors)


if __name__ == "__main__":
    unittest.main()
