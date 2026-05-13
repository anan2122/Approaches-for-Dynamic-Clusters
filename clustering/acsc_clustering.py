"""ACSC clustering implementation."""

import numpy as np

from .clustering_utils import MicroCluster, euclidean_distance


class ACSCClustering:
    """Ant Colony Stream Clustering adapter used by the clustering engine."""

    @staticmethod
    def run(
        points: np.ndarray,
        epsilon: float,
        n_comp: int,
        sleep_max: int,
    ) -> np.ndarray:
        if points.size == 0:
            return np.zeros(shape=(0,), dtype=int)

        epsilon = max(float(epsilon), 1e-9)
        min_component_size = max(int(n_comp), 1)
        smooth_factor = max(int(sleep_max), 0)

        effective_epsilon = epsilon * (1.0 + min(smooth_factor, 100) * 0.001)

        rough_clusters = ACSCClustering._assign_points_to_clusters(points, effective_epsilon)
        micro_clusters = ACSCClustering._convert_to_microclusters(rough_clusters)
        micro_clusters = ACSCClustering._ant_sort_microclusters(
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
    ) -> list[list[tuple[int, np.ndarray]]]:
        clusters: list[list[tuple[int, np.ndarray]]] = []
        cluster_summaries: list[MicroCluster] = []

        for point_index, point in enumerate(points):
            best_cluster_index = -1
            best_distance = float("inf")

            for cluster_index, cluster_summary in enumerate(cluster_summaries):
                distance = euclidean_distance(point, cluster_summary.get_center())
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
        clusters: list[list[tuple[int, np.ndarray]]],
    ) -> list[MicroCluster]:
        micro_clusters: list[MicroCluster] = []

        for cluster_points in clusters:
            micro_cluster = MicroCluster()
            for point_index, point in cluster_points:
                micro_cluster.add_point(point, point_index)
            micro_clusters.append(micro_cluster)

        return micro_clusters

    @staticmethod
    def _ant_sort_microclusters(
        microclusters: list[MicroCluster],
        epsilon: float,
        sleep_max: int,
    ) -> list[MicroCluster]:
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

                    distance = euclidean_distance(sparse_center, neighbor_cluster.get_center())
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
