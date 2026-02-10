#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Clustering Analysis Notebook

Performs K-Means clustering with optimal number of clusters and evaluates results.
"""

import numpy as np
from src.clustering_models import KMeansClustering
from src.evaluation_metrics import ClusteringEvaluator
from src.data_preprocessing import DataPreprocessor
import matplotlib.pyplot as plt

# Load and preprocess data
preprocessor = DataPreprocessor(scaling_method='standard')
X_processed = preprocessor.fit_transform(pd.read_csv('data/raw/customers.csv'))

# Find optimal clusters using Elbow method
inertias = []
silhouette_scores = []
K_range = range(2, 11)

for k in K_range:
    kmeans = KMeansClustering(n_clusters=k)
    labels = kmeans.fit(X_processed)
    inertias.append(kmeans.model.inertia_)
    silhouette_scores.append(silhouette_score(X_processed, labels))

# Plot Elbow curve
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(K_range, inertias, 'bo-')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.title('Elbow Method')

plt.subplot(1, 2, 2)
plt.plot(K_range, silhouette_scores, 'ro-')
plt.xlabel('Number of Clusters')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Score')
plt.tight_layout()
plt.savefig('visualizations/clustering_analysis.png', dpi=300)

# Train final model with optimal clusters (k=5)
optimal_k = 5
kmeans = KMeansClustering(n_clusters=optimal_k)
labels = kmeans.fit(X_processed)

# Evaluate
evaluator = ClusteringEvaluator()
metrics = evaluator.evaluate(X_processed, labels)
print(f"Clustering Metrics for k={optimal_k}:")
for metric, value in metrics.items():
    print(f"  {metric}: {value:.4f}")

print(f"\nCluster distribution:")
for cluster in range(optimal_k):
    count = np.sum(labels == cluster)
    percentage = (count / len(labels)) * 100
    print(f"  Cluster {cluster}: {count} samples ({percentage:.1f}%)")
