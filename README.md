# Evaluation of Approaches for Identification of Dynamic Clusters
## Overview
This project evaluates multiple clustering approaches for identifying dynamic clusters in continuously arriving 3D spatial data streams. The system compares DBSCAN, ACSC (Ant Colony Stream Clustering), and DenStream-based stream clustering techniques using metrics such as cluster formation, outlier detection, cluster stability, and execution time.

## Features
- Dynamic stream clustering
- 3D point visualization
- ACSC implementation using micro-clusters
- DenStream implementation with fading and pruning
- DBSCAN baseline comparison
- Cluster evolution tracking
- Performance evaluation across timestamps
- Outlier detection
- Per-timestamp execution time measurement
- Synthetic dataset generation

---

## Technologies Used
- Python
- NumPy
- Scikit-learn
- PyQt 
- Matplotlib

---
## Algorithms Used

### ACSC (Ant Colony Stream Clustering)
The ACSC algorithm:
1. Creates rough clusters
2. Converts them into micro-clusters
3. Computes:
   - LS (Linear Sum)
   - SS (Squared Sum)
   - Radius
4. Performs ant-inspired merging
5. Detects outliers

### DBSCAN
DBSCAN groups points based on density using:
- epsilon (eps)
- minimum samples (min_samples)

### DenStream

The DenStream-inspired implementation:

1. Maintains Potential Micro-Clusters (PMC)
2. Maintains Outlier Micro-Clusters (OMC)
3. Applies fading functions to old data
4. Promotes dense outlier clusters
5. Prunes weak clusters
6. Supports evolving data streams

---

## ACSC Formulae

### Cluster Center
Center = LS / N

### Radius
r = sqrt(SS/N - (LS/N)^2)

### DenStream Fading Function

f(t) = 2^(-λt)

where:

- λ = decay factor
- t = elapsed time

The fading function reduces the influence of older data points over time.

---

## System Architecture

```text
Streaming Data
      │
      ▼
Data Generator
      │
      ▼
Clustering Engine
 ├── DBSCAN
 ├── ACSC
 └── DenStream
      │
      ▼
Performance Evaluation
      │
      ▼
PyQt Visualization
```

---
## Experimental Evaluation

The algorithms were evaluated on synthetic 3D streaming datasets across 100 timestamps.

Evaluation metrics included:

- Cluster Formation
- Outlier Detection
- Cluster Stability
- Execution Time
- Adaptability to Dynamic Data Streams

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run Application

```bash
python main.py
```
