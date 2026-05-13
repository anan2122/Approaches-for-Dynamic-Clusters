"""Data loading utilities for clustering visualization app."""

import re
from typing import List

import numpy as np


class DataLoader:
    """Loads and parses input data files into timestamp-based point arrays."""

    # Matches tuples such as: (1,2,3), (1.5, -2.3, 0), etc.
    _POINT_PATTERN = re.compile(
        r"\(\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*,\s*"
        r"([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\s*\)"
    )

    def load_points(self, file_path: str) -> List[np.ndarray]:
        """Read file rows and extract 3D tuples per timestamp line.

        Expected row format:
            timestamp, (x,y,z), (x,y,z), ..., (x,y,z)n
        """
        timestamp_points: List[np.ndarray] = []

        with open(file_path, "r", encoding="utf-8") as file_obj:
            for raw_line in file_obj:
                line = raw_line.strip()
                if not line:
                    continue

                # Timestamp prefix is ignored; each row becomes one timestamp array.
                matches = self._POINT_PATTERN.findall(line)
                points: List[List[float]] = []
                for x_val, y_val, z_val in matches:
                    points.append([float(x_val), float(y_val), float(z_val)])
                timestamp_points.append(np.asarray(points, dtype=float).reshape(-1, 3))

        return timestamp_points
