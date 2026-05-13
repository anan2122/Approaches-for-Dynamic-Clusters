"""Generate synthetic streaming clustering dataset.

Produces 100 timestamps of 3D points and writes to `synthetic_data.txt` in the
format: timestamp,(x,y,z),(x,y,z),...

Behavior:
- 100 timestamps
- ~40-60 points per timestamp (random)
- 2-3 moving clusters whose centers drift smoothly over time
- gaussian noise around cluster centers
- occasional random outliers and occasional cluster add/remove events
"""

import numpy as np
from pathlib import Path

# Reproducible randomness
np.random.seed(42)

# Config
num_timestamps = 100
min_points_per_ts = 40
max_points_per_ts = 60
initial_num_clusters = 2
max_clusters = 4
outlier_prob = 0.15  # chance per point to be an outlier
cluster_evolution_prob = 0.05  # chance to add/remove a cluster each timestamp

# Initialize cluster centers and smooth drift vectors
cluster_centers: list[np.ndarray] = [
    np.array([2.0, 2.0, 2.0]),
    np.array([8.0, 8.0, 8.0]),
]

# Per-cluster drift directions (small, smooth movements)
cluster_drifts: list[np.ndarray] = [
    np.array([0.02, 0.015, -0.01]),
    np.array([-0.015, 0.02, 0.01]),
]

output_lines: list[str] = []

for t in range(1, num_timestamps + 1):
    # Occasionally add or remove a cluster to simulate concept evolution
    if np.random.rand() < cluster_evolution_prob:
        if np.random.rand() < 0.5 and len(cluster_centers) > 1:
            # remove a random cluster
            idx = np.random.randint(0, len(cluster_centers))
            del cluster_centers[idx]
            del cluster_drifts[idx]
        elif len(cluster_centers) < max_clusters:
            # add a new cluster near an existing one
            anchor = cluster_centers[np.random.randint(0, len(cluster_centers))]
            new_center = anchor + np.random.normal(0, 1.0, 3)
            cluster_centers.append(new_center)
            cluster_drifts.append(np.random.normal(0, 0.02, 3))

    # Move cluster centers smoothly
    for i in range(len(cluster_centers)):
        cluster_centers[i] = cluster_centers[i] + cluster_drifts[i] + np.random.normal(0, 0.005, 3)

    # Number of points for this timestamp
    n_points = int(np.random.randint(min_points_per_ts, max_points_per_ts + 1))

    # Decide number of outliers (some timestamps have more outliers)
    max_outliers = max(1, int(n_points * 0.12))
    n_outliers = int(np.random.binomial(max_outliers, 0.5))

    # Allocate remaining points to clusters
    n_clusters = len(cluster_centers)
    n_cluster_points = n_points - n_outliers
    if n_cluster_points < n_clusters:
        # ensure at least one point per cluster when possible
        n_outliers = max(0, n_points - n_clusters)
        n_cluster_points = n_points - n_outliers

    # Random split of points across clusters
    if n_clusters > 0:
        proportions = np.random.dirichlet(np.ones(n_clusters))
        raw_counts = (proportions * n_cluster_points).astype(int)
        # Fix rounding to match total
        diff = n_cluster_points - raw_counts.sum()
        for i in range(diff):
            raw_counts[i % n_clusters] += 1
        cluster_point_counts = raw_counts.tolist()
    else:
        cluster_point_counts = []

    points: list[np.ndarray] = []

    # Generate cluster points with gaussian noise
    for idx, count in enumerate(cluster_point_counts):
        center = cluster_centers[idx]
        # std deviation scales moderately with cluster index for variety
        std = 0.25 + 0.05 * (idx % 3)
        cluster_pts = center + np.random.normal(0, std, size=(count, 3))
        points.extend(cluster_pts)

    # Add drifting transitional points to allow points to move between clusters
    n_transitional = max(1, int(0.03 * n_points))
    for _ in range(n_transitional):
        if len(cluster_centers) >= 2:
            a, b = np.random.choice(len(cluster_centers), size=2, replace=False)
            alpha = np.random.rand()
            pt = cluster_centers[a] * (1 - alpha) + cluster_centers[b] * alpha
            pt = pt + np.random.normal(0, 0.15, 3)
            points.append(pt)

    # Add outliers (uniformly in a bounding box around clusters)
    if n_outliers > 0:
        # bounding box from current centers
        centers_stack = np.vstack(cluster_centers)
        mins = centers_stack.min(axis=0) - 3.0
        maxs = centers_stack.max(axis=0) + 3.0
        outliers = np.random.uniform(mins, maxs, size=(n_outliers, 3))
        points.extend(outliers)

    # If we generated too many/few due to rounding, trim or pad with outliers
    if len(points) > n_points:
        points = points[:n_points]
    while len(points) < n_points:
        # pad with small-noise points near a random cluster
        if cluster_centers:
            c = cluster_centers[np.random.randint(0, len(cluster_centers))]
            points.append(c + np.random.normal(0, 0.3, 3))
        else:
            points.append(np.random.uniform(-5, 15, 3))

    # Format and save line
    point_strs = [f"({float(p[0]):.2f},{float(p[1]):.2f},{float(p[2]):.2f})" for p in points]
    line = f"{t}," + ",".join(point_strs)
    output_lines.append(line)

# Persist to file
output_path = Path(__file__).resolve().parent / "synthetic_data.txt"
output_path.write_text("\n".join(output_lines) + "\n", encoding="utf-8")

print(f"✓ Generated synthetic dataset with {num_timestamps} timestamps")
print(f"✓ Saved to: {output_path}")
