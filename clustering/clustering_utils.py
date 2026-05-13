"""Shared clustering utilities and data structures."""

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np


@dataclass
class ClusteringResult:
    """Container for clustering output and derived statistics."""

    labels: np.ndarray
    stats: Dict[str, int]


class MicroCluster:
    """Aggregate 3D points for ACSC-style micro-clustering."""

    def __init__(self) -> None:
        self.N = 0
        self.LS = np.zeros(3, dtype=float)
        self.SS = np.zeros(3, dtype=float)
        self.points: list[np.ndarray] = []
        self.point_indices: list[int] = []

    def add_point(self, point: np.ndarray, point_index: int | None = None) -> None:
        point_array = np.asarray(point, dtype=float).reshape(3)
        self.points.append(point_array)
        if point_index is not None:
            self.point_indices.append(int(point_index))
        self.N += 1
        self.LS += point_array
        self.SS += point_array**2

    def get_center(self) -> np.ndarray:
        if self.N == 0:
            return np.zeros(3, dtype=float)
        return self.LS / self.N

    def get_radius(self) -> float:
        if self.N == 0:
            return 0.0

        center = self.get_center()
        radius_vector = self.SS / self.N - center**2
        radius_vector = np.maximum(radius_vector, 0.0)
        return float(np.sqrt(radius_vector.sum()))


def euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    return float(np.linalg.norm(point_a - point_b))


def coerce_float(value: object, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def coerce_int(value: object, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def placeholder_labels(points: np.ndarray) -> np.ndarray:
    return np.zeros(shape=(points.shape[0],), dtype=int)


def build_stats(labels: np.ndarray) -> Dict[str, int]:
    unique_labels = sorted(set(labels.tolist()))
    outlier_count = int((labels == -1).sum())

    cluster_sizes = []
    for cluster_id in unique_labels:
        if cluster_id == -1:
            continue
        size = int((labels == cluster_id).sum())
        cluster_sizes.append(size)

    total_clusters = len(cluster_sizes)
    target_cluster_size = max(cluster_sizes) if cluster_sizes else 0
    smallest_cluster_size = min(cluster_sizes) if cluster_sizes else 0

    return {
        "total_clusters": total_clusters,
        "target_cluster_size": target_cluster_size,
        "smallest_cluster_size": smallest_cluster_size,
        "outliers": outlier_count,
        "changed_cluster_points": 0,
    }


def count_changed_cluster_points(
    current_points: np.ndarray,
    current_labels: np.ndarray,
    previous_state: Tuple[np.ndarray, np.ndarray] | None,
) -> int:
    if previous_state is None:
        return 0

    previous_points, previous_labels = previous_state
    if current_points.size == 0 or previous_points.size == 0:
        return 0

    if current_points.shape[0] == previous_points.shape[0]:
        matched_previous_labels = previous_labels
    else:
        distances = np.linalg.norm(
            current_points[:, np.newaxis, :] - previous_points[np.newaxis, :, :],
            axis=2,
        )
        nearest_previous_indices = np.argmin(distances, axis=1)
        matched_previous_labels = previous_labels[nearest_previous_indices]

    return int(np.count_nonzero(current_labels != matched_previous_labels))


def validate_clustering_result(
    points: np.ndarray,
    labels: np.ndarray,
    stats: Dict[str, int],
) -> None:
    if len(labels) != len(points):
        error_msg = (
            f"Label/point count mismatch: {len(labels)} labels for {len(points)} points. "
            "Clustering output is invalid."
        )
        print(f"[ERROR] {error_msg}")
        raise ValueError(error_msg)

    for stat_key in ["total_clusters", "outliers"]:
        stat_value = stats.get(stat_key, 0)
        if stat_value < 0:
            error_msg = (
                f"Invalid stat '{stat_key}'={stat_value}. "
                "Statistics must be non-negative."
            )
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

    total_points_in_clusters = sum(
        (labels == cluster_id).sum()
        for cluster_id in set(labels.tolist())
    )
    if total_points_in_clusters != len(points):
        debug_msg = (
            f"[DEBUG] Points distribution warning: {total_points_in_clusters} points "
            f"accounted for in stats vs {len(points)} input points. "
            f"Outliers: {stats.get('outliers', 0)}"
        )
        print(debug_msg)

    print(
        f"[DEBUG] Output validation passed: "
        f"points={len(points)}, labels={len(labels)}, "
        f"clusters={stats.get('total_clusters', 0)}, "
        f"outliers={stats.get('outliers', 0)}"
    )
