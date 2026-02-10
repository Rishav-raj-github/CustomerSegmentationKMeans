"""Customer Segmentation K-Means clustering module.

This module provides a comprehensive implementation of K-Means clustering
for customer segmentation and behavioral analysis.
"""

__version__ = "1.0.0"
__author__ = "Rishav Raj"
__description__ = "Advanced ML project for customer segmentation using K-Means clustering"

from .data_preprocessing import DataPreprocessor
from .clustering_models import KMeansClustering, DBSCANClustering
from .evaluation_metrics import ClusteringEvaluator
from .model_trainer import KMeansTrainer

__all__ = [
    'DataPreprocessor',
    'KMeansClustering',
    'DBSCANClustering',
    'ClusteringEvaluator',
    'KMeansTrainer',
]
