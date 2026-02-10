"""Clustering algorithms implementation module."""
import numpy as np
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import logging

logger = logging.getLogger(__name__)


class KMeansClustering:
    """K-Means clustering implementation."""

    def __init__(self, n_clusters=5, random_state=42, max_iter=300):
        """Initialize K-Means clustering.
        
        Args:
            n_clusters (int): Number of clusters
            random_state (int): Random seed
            max_iter (int): Maximum iterations
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.max_iter = max_iter
        self.model = None
        self.labels = None
        logger.info(f"KMeansClustering initialized with n_clusters={n_clusters}")

    def fit(self, X):
        """Fit K-Means model to data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Cluster labels
        """
        self.model = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            max_iter=self.max_iter,
            n_init=10
        )
        self.labels = self.model.fit_predict(X)
        logger.info(f"K-Means model fitted successfully")
        return self.labels

    def predict(self, X):
        """Predict cluster labels for new data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Predicted cluster labels
        """
        return self.model.predict(X)

    def get_centroids(self):
        """Get cluster centroids.
        
        Returns:
            np.ndarray: Centroid coordinates
        """
        return self.model.cluster_centers_


class DBSCANClustering:
    """DBSCAN clustering implementation."""

    def __init__(self, eps=0.5, min_samples=5):
        """Initialize DBSCAN clustering.
        
        Args:
            eps (float): Neighborhood radius
            min_samples (int): Minimum samples per cluster
        """
        self.eps = eps
        self.min_samples = min_samples
        self.model = None
        self.labels = None
        logger.info(f"DBSCANClustering initialized with eps={eps}")

    def fit(self, X):
        """Fit DBSCAN model to data.
        
        Args:
            X (np.ndarray): Feature matrix
            
        Returns:
            np.ndarray: Cluster labels
        """
        self.model = DBSCAN(eps=self.eps, min_samples=self.min_samples)
        self.labels = self.model.fit_predict(X)
        logger.info(f"DBSCAN model fitted successfully")
        return self.labels
