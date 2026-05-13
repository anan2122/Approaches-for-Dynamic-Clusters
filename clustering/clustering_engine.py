"""Core clustering engine that routes to algorithm modules."""

from typing import Any, Dict, Tuple

import numpy as np

from .acsc_clustering import ACSCClustering
from .clustering_utils import (
    ClusteringResult,
    build_stats,
    coerce_float,
    coerce_int,
    count_changed_cluster_points,
    placeholder_labels,
    validate_clustering_result,
)
from .dbscan_clustering import DBSCANClustering


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
        self._previous_timestamp_state: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

    def run_algorithm(
        self,
        algorithm_name: str,
        points: np.ndarray,
        metric_name: str,
        params: Dict[str, Any] | None = None,
        **legacy_kwargs: Any,
    ) -> ClusteringResult:
        effective_params: Dict[str, Any] = dict(params or {})

        legacy_param_map = legacy_kwargs.get("algorithm_parameters")
        if isinstance(legacy_param_map, dict):
            effective_params.update(legacy_param_map)

        if algorithm_name == self.ALGO_DBSCAN:
            eps = coerce_float(
                effective_params.get("eps", legacy_kwargs.get("dbscan_eps")),
                default=0.5,
            )
            min_samples = coerce_int(
                effective_params.get("min_samples", legacy_kwargs.get("dbscan_min_samples")),
                default=5,
            )
            labels = DBSCANClustering.run(
                points=points,
                metric_name=metric_name,
                eps=eps,
                min_samples=min_samples,
            )
        elif algorithm_name == self.ALGO_ACSC:
            epsilon = coerce_float(effective_params.get("epsilon"), default=0.5)
            n_comp = coerce_int(effective_params.get("nComp"), default=5)
            sleep_max = coerce_int(effective_params.get("sleepMax"), default=10)
            labels = ACSCClustering.run(
                points=points,
                epsilon=epsilon,
                n_comp=n_comp,
                sleep_max=sleep_max,
            )
        else:
            labels = placeholder_labels(points)

        stats = build_stats(labels)
        previous_state = self._previous_timestamp_state.get(algorithm_name)
        stats["changed_cluster_points"] = count_changed_cluster_points(
            current_points=points,
            current_labels=labels,
            previous_state=previous_state,
        )
        self._previous_timestamp_state[algorithm_name] = (
            np.asarray(points, dtype=float).copy(),
            np.asarray(labels, dtype=int).copy(),
        )
        validate_clustering_result(points, labels, stats)
        return ClusteringResult(labels=labels, stats=stats)
