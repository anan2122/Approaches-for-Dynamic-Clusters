"""DenStream clustering implementation for streaming 3D points."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clustering_utils import euclidean_distance


@dataclass
class MicroCluster:
    """Base DenStream micro-cluster with fading and CF statistics."""

    cf1: np.ndarray
    cf2: np.ndarray
    weight: float
    last_update_time: int
    creation_time: int

    @classmethod
    def from_point(cls, point: np.ndarray, timestamp: int) -> "MicroCluster":
        point_array = np.asarray(point, dtype=float).reshape(3)
        return cls(
            cf1=point_array.copy(),
            cf2=point_array**2,
            weight=1.0,
            last_update_time=timestamp,
            creation_time=timestamp,
        )

    @staticmethod
    def fading_factor(decay_lambda: float, delta_t: int) -> float:
        """Return DenStream fading factor f(t) = 2^(-lambda * delta_t)."""
        return float(2.0 ** (-decay_lambda * max(int(delta_t), 0)))

    def fade_by_elapsed_time(self, delta_t: int, decay_lambda: float) -> None:
        """Fade CF statistics and weight by the elapsed timestamp delta."""
        factor = self.fading_factor(decay_lambda, delta_t)
        self.cf1 *= factor
        self.cf2 *= factor
        self.weight *= factor

    def apply_fading(self, current_time: int, decay_lambda: float) -> None:
        """Fade CF statistics and weight forward in time."""
        delta_t = max(current_time - self.last_update_time, 0)
        self.fade_by_elapsed_time(delta_t, decay_lambda)
        self.last_update_time = current_time

    def add_point(self, point: np.ndarray, current_time: int, decay_lambda: float) -> None:
        """Apply fading, then absorb a new point."""
        self.apply_fading(current_time, decay_lambda)
        point_array = np.asarray(point, dtype=float).reshape(3)
        self.cf1 += point_array
        self.cf2 += point_array**2
        self.weight += 1.0

    def centroid(self) -> np.ndarray:
        if self.weight <= 0.0:
            return np.zeros(3, dtype=float)
        return self.cf1 / self.weight

    def radius_per_dimension(self) -> np.ndarray:
        """Return dimension-wise radius for 3D points using DenStream CF formula.

        r_d = sqrt((CF2_d / w) - (CF1_d / w)^2)
        """
        if self.weight <= 0.0:
            return np.zeros(3, dtype=float)

        mean_per_dim = self.cf1 / self.weight
        variance_per_dim = self.cf2 / self.weight - mean_per_dim**2
        variance_per_dim = np.maximum(variance_per_dim, 0.0)
        return np.sqrt(variance_per_dim)

    def radius(self) -> float:
        """Return scalar cluster radius from the 3D dimension-wise radii."""
        if self.weight <= 0.0:
            return 0.0

        radius_dims = self.radius_per_dimension()
        return float(np.linalg.norm(radius_dims))

    def copy(self) -> "MicroCluster":
        return MicroCluster(
            cf1=self.cf1.copy(),
            cf2=self.cf2.copy(),
            weight=float(self.weight),
            last_update_time=int(self.last_update_time),
            creation_time=int(self.creation_time),
        )


class PotentialMicroCluster(MicroCluster):
    """Potential micro-cluster (p-micro-cluster)."""


class OutlierMicroCluster(MicroCluster):
    """Outlier micro-cluster (o-micro-cluster)."""


class DenStreamEngine:
    """Incremental DenStream engine for 3D streaming data."""

    def __init__(
        self,
        decay_lambda: float = 0.01,
        epsilon: float = 0.5,
        beta: float = 0.2,
        mu: float = 5.0,
        tp: int = 20,
    ) -> None:
        self.decay_lambda = max(float(decay_lambda), 1e-9)
        self.epsilon = max(float(epsilon), 1e-9)
        self.beta = max(float(beta), 1e-6)
        self.mu = max(float(mu), 1.0)
        self.tp = max(int(tp), 1)

        self.timestamp = 0
        self._p_microclusters: list[PotentialMicroCluster] = []
        self._o_microclusters: list[OutlierMicroCluster] = []

    @property
    def potential_microclusters(self) -> list[PotentialMicroCluster]:
        return self._p_microclusters

    @property
    def outlier_microclusters(self) -> list[OutlierMicroCluster]:
        return self._o_microclusters

    def process_points(self, points: np.ndarray) -> np.ndarray:
        """Process a batch of points incrementally and return labels for the batch."""
        points_array = np.asarray(points, dtype=float)
        if points_array.size == 0:
            return np.zeros(shape=(0,), dtype=int)
        if points_array.ndim != 2 or points_array.shape[1] != 3:
            raise ValueError("DenStream expects points with shape (N, 3).")

        for point in points_array:
            self.timestamp += 1
            self._process_single_point(point)

            if self.timestamp % self.tp == 0:
                self._periodic_fading_and_pruning()

        return self._label_points(points_array)

    def _process_single_point(self, point: np.ndarray) -> None:
        """DenStream workflow for one point.

        1) Try nearest p-micro-cluster
        2) If failed, try nearest o-micro-cluster
        3) If failed, create new o-micro-cluster
        4) Periodic fading/pruning handled by caller
        """
        if self._try_merge_into_p_microcluster(point):
            return

        promoted = self._try_merge_into_o_microcluster(point)
        if promoted:
            return

        self._o_microclusters.append(
            OutlierMicroCluster.from_point(point, self.timestamp)
        )

    def _try_merge_into_p_microcluster(self, point: np.ndarray) -> bool:
        nearest = self._nearest_cluster(point, self._p_microclusters)
        if nearest is None:
            return False

        cluster_index, cluster = nearest
        if self._merge_preserves_radius(cluster, point):
            self._p_microclusters[cluster_index].add_point(
                point,
                current_time=self.timestamp,
                decay_lambda=self.decay_lambda,
            )
            return True

        return False

    def _try_merge_into_o_microcluster(self, point: np.ndarray) -> bool:
        nearest = self._nearest_cluster(point, self._o_microclusters)
        if nearest is None:
            return False

        cluster_index, cluster = nearest
        if not self._merge_preserves_radius(cluster, point):
            return False

        cluster.add_point(
            point,
            current_time=self.timestamp,
            decay_lambda=self.decay_lambda,
        )

        if cluster.weight >= self.beta * self.mu:
            self._promote_outlier_microcluster(cluster_index)

        return True

    def _promote_outlier_microcluster(self, cluster_index: int) -> None:
        """Promote a qualifying o-micro-cluster to a p-micro-cluster."""
        cluster = self._o_microclusters[cluster_index]
        promoted_cluster = PotentialMicroCluster(
            cf1=cluster.cf1.copy(),
            cf2=cluster.cf2.copy(),
            weight=float(cluster.weight),
            last_update_time=int(cluster.last_update_time),
            creation_time=int(cluster.creation_time),
        )
        self._p_microclusters.append(promoted_cluster)
        del self._o_microclusters[cluster_index]

    def _nearest_cluster(
        self,
        point: np.ndarray,
        clusters: list[MicroCluster],
    ) -> tuple[int, MicroCluster] | None:
        if not clusters:
            return None

        best_index = -1
        best_distance = float("inf")
        for index, cluster in enumerate(clusters):
            center = cluster.centroid()
            distance = euclidean_distance(point, center)
            if distance < best_distance:
                best_distance = distance
                best_index = index

        if best_index < 0:
            return None
        return best_index, clusters[best_index]

    def _merge_preserves_radius(self, cluster: MicroCluster, point: np.ndarray) -> bool:
        trial_cluster = cluster.copy()
        trial_cluster.add_point(
            point,
            current_time=self.timestamp,
            decay_lambda=self.decay_lambda,
        )
        return trial_cluster.radius() <= self.epsilon

    def _periodic_fading_and_pruning(self) -> None:
        for cluster in self._p_microclusters:
            cluster.apply_fading(self.timestamp, self.decay_lambda)
        for cluster in self._o_microclusters:
            cluster.apply_fading(self.timestamp, self.decay_lambda)

        self._promote_and_prune_microclusters()

    def _promote_and_prune_microclusters(self) -> None:
        """Promote strong o-micro-clusters and prune weak clusters after fading."""
        p_min_weight = self.beta * self.mu
        o_min_weight = 0.5 * self.beta * self.mu

        promoted_indices = [
            index for index, cluster in enumerate(self._o_microclusters) if cluster.weight >= p_min_weight
        ]
        for index in reversed(promoted_indices):
            self._promote_outlier_microcluster(index)

        self._p_microclusters = [
            cluster for cluster in self._p_microclusters if cluster.weight >= p_min_weight
        ]
        self._o_microclusters = [
            cluster for cluster in self._o_microclusters if cluster.weight >= o_min_weight
        ]

    def _label_points(self, points: np.ndarray) -> np.ndarray:
        """Assign each point to the nearest p-micro-cluster if within epsilon."""
        labels = np.full(shape=(points.shape[0],), fill_value=-1, dtype=int)
        if not self._p_microclusters:
            return labels

        centers = np.asarray([cluster.centroid() for cluster in self._p_microclusters])
        for point_index, point in enumerate(points):
            distances = np.linalg.norm(centers - point, axis=1)
            nearest_index = int(np.argmin(distances))
            if float(distances[nearest_index]) <= self.epsilon:
                labels[point_index] = nearest_index

        return labels
