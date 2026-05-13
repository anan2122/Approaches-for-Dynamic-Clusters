# Approaches-for-Dynamic-Clusters
Evaluation of approaches for identification of dynamic clusters.
This project implements ACSC and DBSCAN for clustering dynamic 3D streaming data.

## Features
- Dynamic stream clustering
- 3D point visualization
- ACSC implementation using micro-clusters
- DBSCAN comparison
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

---

## ACSC Formulae

### Cluster Center
Center = LS / N

### Radius
r = sqrt(SS/N - (LS/N)^2)

---

## How to Run

### Install dependencies
```bash
pip install -r requirements.txt
