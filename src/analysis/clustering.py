"""Clustering analysis for dataset exploration.

This module defines the architecture and contracts for an unsupervised
visual clustering pipeline. It is intentionally decoupled from image loading
and embedding extraction.
"""

from __future__ import annotations

import abc
import math
import statistics
from dataclasses import asdict, dataclass, field
from typing import Iterable, Literal, Optional


ClusteringLevel = Literal["image", "animal"]
ReductionMethod = Literal["none", "pca", "umap"]
ClusteringMethod = Literal["none", "kmeans", "hdbscan"]


@dataclass(slots=True)
class ClusteringConfig:
    """Configuration for the exploratory clustering analysis."""

    analysis_level: ClusteringLevel = "image"
    reduction_method: ReductionMethod = "none"
    clustering_method: ClusteringMethod = "none"
    n_clusters: Optional[int] = None
    random_seed: int = 42
    supervised_exclusions: list[str] = field(default_factory=lambda: ["weight_kg", "sex", "view", "dataset"])
    clustering_params: dict[str, object] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self._validate()

    def _validate(self) -> None:
        if self.analysis_level not in {"image", "animal"}:
            raise ValueError("analysis_level must be 'image' or 'animal'.")

        if self.reduction_method not in {"none", "pca", "umap"}:
            raise ValueError("reduction_method must be 'none', 'pca', or 'umap'.")

        if self.clustering_method not in {"none", "kmeans", "hdbscan"}:
            raise ValueError("clustering_method must be 'none', 'kmeans', or 'hdbscan'.")

        if self.clustering_method == "kmeans" and self.n_clusters is None:
            raise ValueError("n_clusters must be provided for kmeans clustering.")

        if self.random_seed < 0:
            raise ValueError("random_seed must be non-negative.")

        if any(not isinstance(value, str) for value in self.supervised_exclusions):
            raise TypeError("supervised_exclusions must be a list of strings.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class FeatureVector:
    """Represents a feature vector for one image or animal."""

    sample_id: str
    embedding: list[float]
    filename: Optional[str] = None
    animal_id: Optional[str] = None
    metadata: dict[str, str] = field(default_factory=dict)

    def validate(self) -> None:
        if not isinstance(self.sample_id, str) or not self.sample_id.strip():
            raise ValueError("FeatureVector.sample_id must be a non-empty string.")

        if not isinstance(self.embedding, list) or len(self.embedding) == 0:
            raise ValueError("FeatureVector.embedding must be a non-empty list of floats.")

        for value in self.embedding:
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("FeatureVector.embedding must contain only finite numeric values.")

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "embedding": list(self.embedding),
            "filename": self.filename,
            "animal_id": self.animal_id,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ClusterResult:
    """Represents the clustering result for a single sample."""

    sample_id: str
    cluster_id: int | str | None
    is_outlier: bool
    distance_to_centroid: Optional[float] = None
    metadata: dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> dict[str, object]:
        return {
            "sample_id": self.sample_id,
            "cluster_id": self.cluster_id,
            "is_outlier": self.is_outlier,
            "distance_to_centroid": self.distance_to_centroid,
            "metadata": dict(self.metadata),
        }


@dataclass(slots=True)
class ClusteringMetrics:
    """Stores clustering quality metrics."""

    silhouette_score: Optional[float]
    calinski_harabasz_score: Optional[float]
    davies_bouldin_score: Optional[float]
    n_clusters: int
    n_samples: int
    n_outliers: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(slots=True)
class ClusteringReport:
    """Stores the result of clustering analysis."""

    config: ClusteringConfig
    metrics: ClusteringMetrics
    results: list[ClusterResult]
    metadata: dict[str, str]
    features_used: list[str]
    excluded_variables: list[str]
    clustering_method: str

    def to_dict(self) -> dict[str, object]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics.to_dict(),
            "results": [result.to_dict() for result in self.results],
            "metadata": dict(self.metadata),
            "features_used": list(self.features_used),
            "excluded_variables": list(self.excluded_variables),
            "clustering_method": self.clustering_method,
        }


class BaseClusterer(abc.ABC):
    """Abstract clusterer interface."""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config

    @abc.abstractmethod
    def fit_predict(self, feature_matrix: list[list[float]]) -> list[int]:
        raise NotImplementedError


class DummyClusterer(BaseClusterer):
    """Dummy clusterer used for architecture validation."""

    def fit_predict(self, feature_matrix: list[list[float]]) -> list[int]:
        return [0 for _ in feature_matrix]


class ClusteringAnalyzer:
    """Performs clustering analysis on feature vectors."""

    def __init__(self, config: ClusteringConfig) -> None:
        self.config = config
        self._validate_config()

    def _validate_config(self) -> None:
        self.config._validate()

    def analyze(self, feature_vectors: list[FeatureVector]) -> ClusteringReport:
        if not isinstance(feature_vectors, list):
            raise TypeError("feature_vectors must be a list of FeatureVector instances.")

        if not feature_vectors:
            raise ValueError("At least one FeatureVector is required for clustering.")

        for feature_vector in feature_vectors:
            if not isinstance(feature_vector, FeatureVector):
                raise TypeError("All items in feature_vectors must be FeatureVector instances.")
            feature_vector.validate()

        feature_matrix = self._build_feature_matrix(feature_vectors)
        cluster_labels = self._cluster(feature_matrix)
        results = self._build_results(feature_vectors, cluster_labels, feature_matrix)
        metrics = self._compute_metrics(feature_matrix, cluster_labels, results)

        return ClusteringReport(
            config=self.config,
            metrics=metrics,
            results=results,
            metadata={
                "description": "Exploratory unsupervised visual clustering report.",
                "objective": "Discover natural structure without using supervised targets.",
                "note": (
                    "This result is an exploratory clustering analysis architecture. "
                    "When clustering_method='none', no algorithmic clustering has been executed "
                    "and the report should not be interpreted as a scientific clustering outcome."
                ),
            },
            features_used=["embedding"],
            excluded_variables=list(self.config.supervised_exclusions),
            clustering_method=self.config.clustering_method,
        )

    def _build_feature_matrix(self, feature_vectors: list[FeatureVector]) -> list[list[float]]:
        return [list(vector.embedding) for vector in feature_vectors]

    def _cluster(self, feature_matrix: list[list[float]]) -> list[int]:
        if self.config.clustering_method == "none":
            clusterer = DummyClusterer(self.config)
        elif self.config.clustering_method == "kmeans":
            raise NotImplementedError("K-Means clustering is not implemented in this version.")
        elif self.config.clustering_method == "hdbscan":
            raise NotImplementedError("HDBSCAN clustering is not implemented in this version.")
        else:
            raise ValueError(f"Unsupported clustering_method: {self.config.clustering_method}")

        return clusterer.fit_predict(feature_matrix)

    def _build_results(
        self,
        feature_vectors: list[FeatureVector],
        cluster_labels: list[int],
        feature_matrix: list[list[float]],
    ) -> list[ClusterResult]:
        centroids = self._compute_centroids(feature_matrix, cluster_labels)

        results: list[ClusterResult] = []
        for vector, label in zip(feature_vectors, cluster_labels):
            distance = self._distance_to_centroid(vector.embedding, centroids[label]) if label in centroids else None
            results.append(
                ClusterResult(
                    sample_id=vector.sample_id,
                    cluster_id=label,
                    is_outlier=False,
                    distance_to_centroid=distance,
                    metadata={"filename": vector.filename or "", "animal_id": vector.animal_id or ""},
                )
            )

        return results

    def _compute_centroids(self, feature_matrix: list[list[float]], labels: list[int]) -> dict[int, list[float]]:
        centroids: dict[int, list[float]] = {}
        counts: dict[int, int] = {}

        for row, label in zip(feature_matrix, labels):
            if label not in centroids:
                centroids[label] = [0.0] * len(row)
                counts[label] = 0

            for index, value in enumerate(row):
                centroids[label][index] += value
            counts[label] += 1

        for label, centroid in centroids.items():
            count = counts[label]
            centroids[label] = [value / count for value in centroid]

        return centroids

    def _distance_to_centroid(self, embedding: list[float], centroid: list[float]) -> float:
        return math.sqrt(sum((value - centroid[i]) ** 2 for i, value in enumerate(embedding)))

    def _compute_metrics(
        self,
        feature_matrix: list[list[float]],
        labels: list[int],
        results: list[ClusterResult],
    ) -> ClusteringMetrics:
        n_samples = len(feature_matrix)
        unique_labels = set(labels)
        n_clusters = len(unique_labels)
        n_outliers = sum(1 for result in results if result.is_outlier)

        if n_clusters <= 1:
            return ClusteringMetrics(
                silhouette_score=None,
                calinski_harabasz_score=None,
                davies_bouldin_score=None,
                n_clusters=n_clusters,
                n_samples=n_samples,
                n_outliers=n_outliers,
            )

        silhouette = self._silhouette_score(feature_matrix, labels)
        calinski_harabasz = self._calinski_harabasz_score(feature_matrix, labels)
        davies_bouldin = self._davies_bouldin_score(feature_matrix, labels)

        return ClusteringMetrics(
            silhouette_score=silhouette,
            calinski_harabasz_score=calinski_harabasz,
            davies_bouldin_score=davies_bouldin,
            n_clusters=n_clusters,
            n_samples=n_samples,
            n_outliers=n_outliers,
        )

    def _silhouette_score(self, feature_matrix: list[list[float]], labels: list[int]) -> Optional[float]:
        distances = self._pairwise_distances(feature_matrix)
        label_list = list(set(labels))
        if len(label_list) <= 1:
            return None

        silhouettes: list[float] = []
        for index, label in enumerate(labels):
            same_cluster_distances = [distances[index][j] for j, other_label in enumerate(labels) if other_label == label and j != index]
            other_cluster_distances = []
            for other_label in label_list:
                if other_label == label:
                    continue
                other_cluster_distances.extend([distances[index][j] for j, current_label in enumerate(labels) if current_label == other_label])

            if not same_cluster_distances or not other_cluster_distances:
                continue

            a = statistics.mean(same_cluster_distances)
            b = min(statistics.mean([distances[index][j] for j, current_label in enumerate(labels) if current_label == other_label]) for other_label in label_list if other_label != label)
            silhouettes.append((b - a) / max(a, b))

        return statistics.mean(silhouettes) if silhouettes else None

    def _calinski_harabasz_score(self, feature_matrix: list[list[float]], labels: list[int]) -> Optional[float]:
        if len(set(labels)) <= 1:
            return None

        overall_centroid = [statistics.mean(column) for column in zip(*feature_matrix)]
        cluster_centroids = self._compute_centroids(feature_matrix, labels)
        cluster_sizes: dict[int, int] = {}
        for label in labels:
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        between_dispersion = 0.0
        within_dispersion = 0.0
        for row, label in zip(feature_matrix, labels):
            centroid = cluster_centroids[label]
            within_dispersion += sum((value - centroid[i]) ** 2 for i, value in enumerate(row))
            between_dispersion += sum((centroid[i] - overall_centroid[i]) ** 2 for i in range(len(row))) * cluster_sizes[label]

        n_clusters = len(cluster_centroids)
        return (between_dispersion / (n_clusters - 1)) / (within_dispersion / (len(feature_matrix) - n_clusters)) if within_dispersion > 0 else None

    def _davies_bouldin_score(self, feature_matrix: list[list[float]], labels: list[int]) -> Optional[float]:
        if len(set(labels)) <= 1:
            return None

        cluster_centroids = self._compute_centroids(feature_matrix, labels)
        cluster_sizes: dict[int, int] = {}
        cluster_scatters: dict[int, float] = {}

        for row, label in zip(feature_matrix, labels):
            cluster_sizes[label] = cluster_sizes.get(label, 0) + 1

        for label in cluster_centroids:
            members = [row for row, current_label in zip(feature_matrix, labels) if current_label == label]
            centroid = cluster_centroids[label]
            cluster_scatters[label] = statistics.mean(
                math.sqrt(sum((value - centroid[i]) ** 2 for i, value in enumerate(row)))
                for row in members
            ) if members else 0.0

        ratios: list[float] = []
        labels_set = list(cluster_centroids.keys())
        for i in labels_set:
            max_ratio = 0.0
            for j in labels_set:
                if i == j:
                    continue
                separation = math.sqrt(sum((cluster_centroids[i][d] - cluster_centroids[j][d]) ** 2 for d in range(len(cluster_centroids[i]))))
                if separation == 0:
                    continue
                ratio = (cluster_scatters[i] + cluster_scatters[j]) / separation
                max_ratio = max(max_ratio, ratio)
            ratios.append(max_ratio)

        return statistics.mean(ratios) if ratios else None

    def _pairwise_distances(self, feature_matrix: list[list[float]]) -> list[list[float]]:
        distances: list[list[float]] = []
        for i, row in enumerate(feature_matrix):
            row_distances: list[float] = []
            for j, other_row in enumerate(feature_matrix):
                row_distances.append(math.sqrt(sum((value - other_row[k]) ** 2 for k, value in enumerate(row))))
            distances.append(row_distances)
        return distances
