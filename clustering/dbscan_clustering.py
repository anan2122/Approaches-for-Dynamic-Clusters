"""DBSCAN clustering implementation."""

import numpy as np
from sklearn.cluster import DBSCAN


class DBSCANClustering:
    """DBSCAN adapter used by the clustering engine."""

    METRIC_EUCLIDEAN = "Euclidean"
    METRIC_MANHATTAN = "Manhattan"
    METRIC_SPEED = "Speed-based"
    METRIC_HEADING = "Heading-based"
    METRIC_COMBINATION = "Combination"

    @staticmethod
    def run(
        points: np.ndarray,
        metric_name: str,
        eps: float,
        min_samples: int,
    ) -> np.ndarray:
        metric_map = {
            DBSCANClustering.METRIC_EUCLIDEAN: "euclidean",
            DBSCANClustering.METRIC_MANHATTAN: "manhattan",
            DBSCANClustering.METRIC_SPEED: "euclidean",
            DBSCANClustering.METRIC_HEADING: "euclidean",
            DBSCANClustering.METRIC_COMBINATION: "euclidean",
        }

        dbscan_metric = metric_map.get(metric_name, "euclidean")
        print(
            "[DEBUG] Running DBSCAN with "
            f"eps={eps}, min_samples={min_samples}, metric={dbscan_metric}"
        )
        model = DBSCAN(eps=eps, min_samples=min_samples, metric=dbscan_metric)
        return model.fit_predict(points)
