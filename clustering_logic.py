"""Clustering service layer for running available algorithms."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import numpy as np
from sklearn.cluster import DBSCAN


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
        """Add a 3D point to the micro-cluster."""
        point_array = np.asarray(point, dtype=float).reshape(3)
        self.points.append(point_array)
        if point_index is not None:
            self.point_indices.append(int(point_index))
        self.N += 1
        self.LS += point_array
        self.SS += point_array**2

    def get_center(self) -> np.ndarray:
        """Return the micro-cluster center as LS / N."""
        if self.N == 0:
            return np.zeros(3, dtype=float)
        return self.LS / self.N

    def get_radius(self) -> float:
        """Return the micro-cluster radius using sqrt(SS/N - (LS/N)^2)."""
        if self.N == 0:
            return 0.0

        center = self.get_center()
        radius_vector = self.SS / self.N - center**2
        radius_vector = np.maximum(radius_vector, 0.0)
        return float(np.sqrt(radius_vector.sum()))


def _euclidean_distance(point_a: np.ndarray, point_b: np.ndarray) -> float:
    """Compute Euclidean distance between two 3D points."""
    return float(np.linalg.norm(point_a - point_b))


class ClusteringEngine:
    """Runs clustering algorithms in a pluggable, extensible way."""

    ALGO_DBSCAN = "DBSCAN"
    ALGO_ACSC = "Ant Colony Stream Clustering"
    ALGO_INCREMENTAL_DYNAMIC = "Incremental Dynamic Clustering"
    ALGO_3D_NEIGHBOURHOOD = "3D Neighbourhood Clustering"

    METRIC_EUCLIDEAN = "Euclidean"
    METRIC_MANHATTAN = "Manhattan"
    METRIC_SPEED = "Speed-based"
    METRIC_HEADING = "Heading-based"
    METRIC_COMBINATION = "Combination"

    def __init__(self) -> None:
        # Track the previous timestamp per algorithm so we can compare labels over time.
        self._previous_timestamp_state: Dict[
            str,
            Tuple[np.ndarray, np.ndarray],
        ] = {}

    def run_algorithm(
        self,
        algorithm_name: str,
        points: np.ndarray,
        metric_name: str,
        params: Dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> ClusteringResult:
        """Route request to the corresponding clustering implementation."""
        # Backward compatibility:
        # - current GUI passes algorithm_parameters=...
        # - older callers may still pass dbscan_eps/dbscan_min_samples
        effective_params: Dict[str, Any] = dict(params or {})

        legacy_param_map = legacy_kwargs.get("algorithm_parameters")
        if isinstance(legacy_param_map, dict):
            effective_params.update(legacy_param_map)

        if algorithm_name == self.ALGO_DBSCAN:
            eps = self._coerce_float(
                effective_params.get("eps", legacy_kwargs.get("dbscan_eps")),
                default=0.5,
            )
            min_samples = self._coerce_int(
                effective_params.get(
                    "min_samples",
                    legacy_kwargs.get("dbscan_min_samples"),
                ),
                default=5,
            )
            labels = self._run_dbscan(
                points=points,
                metric_name=metric_name,
                eps=eps,
                min_samples=min_samples,
            )
        elif algorithm_name == self.ALGO_ACSC:
            epsilon = self._coerce_float(
                effective_params.get("epsilon"),
                default=0.5,
            )
            n_comp = self._coerce_int(
                effective_params.get("nComp"),
                default=5,
            )
            sleep_max = self._coerce_int(
                effective_params.get("sleepMax"),
                default=10,
            )
            labels = self._run_acsc(
                points=points,
                epsilon=epsilon,
                n_comp=n_comp,
                sleep_max=sleep_max,
            )
        else:
            # Placeholder output for algorithms not implemented yet.
            labels = self._placeholder_labels(points)

        stats = self._build_stats(labels)
        previous_state = self._previous_timestamp_state.get(algorithm_name)
        stats["changed_cluster_points"] = self._count_changed_cluster_points(
            current_points=points,
            current_labels=labels,
            previous_state=previous_state,
        )
        self._previous_timestamp_state[algorithm_name] = (
            np.asarray(points, dtype=float).copy(),
            np.asarray(labels, dtype=int).copy(),
        )
        self._validate_clustering_result(points, labels, stats)
        return ClusteringResult(labels=labels, stats=stats)

    def _run_dbscan(
        self,
        points: np.ndarray,
        metric_name: str,
        eps: float,
        min_samples: int,
    ) -> np.ndarray:
        """Execute DBSCAN with a simple metric mapping."""
        metric_map = {
            self.METRIC_EUCLIDEAN: "euclidean",
            self.METRIC_MANHATTAN: "manhattan",
            # Placeholder mapping for now; future work can derive custom metrics.
            self.METRIC_SPEED: "euclidean",
            self.METRIC_HEADING: "euclidean",
            self.METRIC_COMBINATION: "euclidean",
        }

        dbscan_metric = metric_map.get(metric_name, "euclidean")
        print(
            "[DEBUG] Running DBSCAN with "
            f"eps={eps}, min_samples={min_samples}, metric={dbscan_metric}"
        )
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=dbscan_metric)
        return model.fit_predict(points)

    def _run_acsc(
        self,
        points: np.ndarray,
        epsilon: float,
        n_comp: int,
        sleep_max: int,
    ) -> np.ndarray:
        """Run a lightweight ACSC-style stream clustering approximation.

        This implementation first performs rough center-based clustering and then
        filters small clusters as noise, producing DBSCAN-compatible labels.
        """
        if points.size == 0:
            return np.zeros(shape=(0,), dtype=int)

        # Clamp parameters to keep behavior stable for invalid inputs.
        epsilon = max(float(epsilon), 1e-9)
        min_component_size = max(int(n_comp), 1)
        smooth_factor = max(int(sleep_max), 0)

        # A small smoothing factor models ACSC temporal settling behavior.
        effective_epsilon = epsilon * (1.0 + min(smooth_factor, 100) * 0.001)

        rough_clusters = self._assign_points_to_clusters(points, effective_epsilon)
        micro_clusters = self._convert_to_microclusters(rough_clusters)
        micro_clusters = self._ant_sort_microclusters(
            microclusters=micro_clusters,
            epsilon=effective_epsilon,
            sleep_max=smooth_factor,
        )
        labels = np.full(shape=(points.shape[0],), fill_value=-1, dtype=int)

        for cluster_id, micro_cluster in enumerate(micro_clusters):
            if micro_cluster.N < min_component_size:
                continue

            labels[np.asarray(micro_cluster.point_indices, dtype=int)] = cluster_id

        print(
            "[DEBUG] Running ACSC with "
            f"epsilon={epsilon}, nComp={n_comp}, sleepMax={sleep_max}, "
            f"effective_epsilon={effective_epsilon}"
        )
        return labels

    @staticmethod
    def _assign_points_to_clusters(
        points: np.ndarray,
        epsilon: float,
    ) -> List[List[tuple[int, np.ndarray]]]:
        """Assign each point to the nearest cluster center or create a new cluster.

        Returns a list of clusters, where each cluster is a list of (index, point) pairs.
        """
        clusters: List[List[tuple[int, np.ndarray]]] = []
        cluster_summaries: List[MicroCluster] = []

        for point_index, point in enumerate(points):
            best_cluster_index = -1
            best_distance = float("inf")

            for cluster_index, cluster_summary in enumerate(cluster_summaries):
                distance = _euclidean_distance(point, cluster_summary.get_center())
                if distance <= epsilon and distance < best_distance:
                    best_distance = distance
                    best_cluster_index = cluster_index

            if best_cluster_index >= 0:
                clusters[best_cluster_index].append((point_index, point))
                cluster_summaries[best_cluster_index].add_point(point, point_index)
            else:
                clusters.append([(point_index, point)])
                new_cluster = MicroCluster()
                new_cluster.add_point(point, point_index)
                cluster_summaries.append(new_cluster)

        return clusters

    @staticmethod
    def _convert_to_microclusters(
        clusters: List[List[tuple[int, np.ndarray]]],
    ) -> List[MicroCluster]:
        """Convert rough point clusters into MicroCluster objects."""
        micro_clusters: List[MicroCluster] = []

        for cluster_points in clusters:
            micro_cluster = MicroCluster()
            for point_index, point in cluster_points:
                micro_cluster.add_point(point, point_index)
            micro_clusters.append(micro_cluster)

        return micro_clusters

    @staticmethod
    def _ant_sort_microclusters(
        microclusters: List[MicroCluster],
        epsilon: float,
        sleep_max: int,
    ) -> List[MicroCluster]:
        """Perform a simplified ant-based sorting and merging phase.

        Sparse microclusters are repeatedly considered for merging with their most
        similar neighbor while the merged radius stays within epsilon.
        """
        if len(microclusters) <= 1 or sleep_max <= 0:
            return microclusters

        clusters = list(microclusters)

        def cluster_density(cluster: MicroCluster) -> float:
            radius = max(cluster.get_radius(), 1e-9)
            return cluster.N / radius

        def merge_if_valid(target: MicroCluster, source: MicroCluster) -> bool:
            merged = MicroCluster()
            for point_index, point in zip(target.point_indices, target.points):
                merged.add_point(point, point_index)
            for point_index, point in zip(source.point_indices, source.points):
                merged.add_point(point, point_index)
            if merged.get_radius() <= epsilon:
                target.points = merged.points
                target.point_indices = merged.point_indices
                target.N = merged.N
                target.LS = merged.LS
                target.SS = merged.SS
                return True
            return False

        for _iteration in range(sleep_max):
            changed = False
            if len(clusters) <= 1:
                break

            densities = [cluster_density(cluster) for cluster in clusters]
            average_density = float(np.mean(densities)) if densities else 0.0

            sparse_indices = [
                index
                for index, density in enumerate(densities)
                if density <= average_density
            ]

            if not sparse_indices:
                break

            sparse_indices.sort(key=lambda index: densities[index])

            for sparse_index in sparse_indices:
                if sparse_index >= len(clusters):
                    continue

                sparse_cluster = clusters[sparse_index]
                if sparse_cluster.N == 0:
                    continue

                best_neighbor_index = -1
                best_distance = float("inf")
                sparse_center = sparse_cluster.get_center()

                for neighbor_index, neighbor_cluster in enumerate(clusters):
                    if neighbor_index == sparse_index or neighbor_cluster.N == 0:
                        continue

                    distance = _euclidean_distance(sparse_center, neighbor_cluster.get_center())
                    if distance < best_distance:
                        best_distance = distance
                        best_neighbor_index = neighbor_index

                if best_neighbor_index < 0:
                    continue

                target_index = min(sparse_index, best_neighbor_index)
                source_index = max(sparse_index, best_neighbor_index)

                if target_index >= len(clusters) or source_index >= len(clusters):
                    continue

                target_cluster = clusters[target_index]
                source_cluster = clusters[source_index]

                if target_cluster is source_cluster:
                    continue

                if merge_if_valid(target_cluster, source_cluster):
                    del clusters[source_index]
                    changed = True
                    break

            if not changed:
                break

        return clusters

    @staticmethod
    def _coerce_float(value: object, default: float) -> float:
        """Safely parse float parameters with fallback defaults."""
        try:
            return float(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _coerce_int(value: object, default: int) -> int:
        """Safely parse int parameters with fallback defaults."""
        try:
            return int(value)
        except (TypeError, ValueError):
            return default

    @staticmethod
    def _placeholder_labels(points: np.ndarray) -> np.ndarray:
        """Return one-cluster labels to keep visualization functional."""
        return np.zeros(shape=(points.shape[0],), dtype=int)

    @staticmethod
    def _build_stats(labels: np.ndarray) -> Dict[str, int]:
        """Compute statistics expected by the GUI panels."""
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

    @staticmethod
    def _count_changed_cluster_points(
        current_points: np.ndarray,
        current_labels: np.ndarray,
        previous_state: Tuple[np.ndarray, np.ndarray] | None,
    ) -> int:
        """Count points whose cluster assignment changed since the prior timestamp."""
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

    @staticmethod
    def _validate_clustering_result(
        points: np.ndarray,
        labels: np.ndarray,
        stats: Dict[str, int],
    ) -> None:
        """Validate clustering output for consistency and correctness.

        Raises ValueError if critical validation checks fail.
        Logs debug info for warnings.
        """
        # Check 1: Label count matches point count
        if len(labels) != len(points):
            error_msg = (
                f"Label/point count mismatch: {len(labels)} labels for {len(points)} points. "
                "Clustering output is invalid."
            )
            print(f"[ERROR] {error_msg}")
            raise ValueError(error_msg)

        # Check 2: Stats values are non-negative
        for stat_key in ["total_clusters", "outliers"]:
            stat_value = stats.get(stat_key, 0)
            if stat_value < 0:
                error_msg = (
                    f"Invalid stat '{stat_key}'={stat_value}. "
                    "Statistics must be non-negative."
                )
                print(f"[ERROR] {error_msg}")
                raise ValueError(error_msg)

        # Check 3: Total points in stats equals input points
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

        # Check 4: Debug log for validation success
        print(
            f"[DEBUG] Output validation passed: "
            f"points={len(points)}, labels={len(labels)}, "
            f"clusters={stats.get('total_clusters', 0)}, "
            f"outliers={stats.get('outliers', 0)}"
        )
