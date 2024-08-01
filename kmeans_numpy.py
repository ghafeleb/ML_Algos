import sys, os
sys.path.insert(0, '../')
import numpy as np
from typing import List

class KMeans_NumPy:
    def __init__(self, k: int, tol: float = 1e-5) -> None:
        self.k = k
        self.centroids = None
        self.tol = tol
     
    
    def stopping_criteria(self, centroids, pre_centroids) -> float:
        dist_sum = 0
        for point1, point2 in zip(centroids, pre_centroids):
            dist_sum += self.get_distance(point1, point2)
        return dist_sum < self.tol
        
    def fit(self, data: np.ndarray) -> None:
        # Initialize centroids
        self.centroids = self.initialize_centroids(data)
        
        # Loop
        while True:
            pre_centroids = self.centroids
            # Get labels
            labels = self.predict(data, )
            # Update clusters
            self.centroids = self.update_centroids(data, labels)
            # Check the stopping criteria
            if self.stopping_criteria(self.centroids, pre_centroids):
                break
            pre_centroids = self.centroids
        
        
    def initialize_centroids(self, data: np.ndarray) -> np.ndarray:
        min_range = np.min(data, axis=0)
        max_range = np.max(data, axis=0)
        centroids = np.random.rand(self.k, data.shape[1])
        centroids = centroids * (max_range - min_range) + min_range
        return centroids
    
    def get_distance(self, x, y) -> float:
        distance = np.sqrt((x - y).T.dot(x - y))
        return distance
    
    def predict(self, data: np.ndarray,) -> List:
        labels = []
        for point in data:
            distances = [self.get_distance(point, centroid) for centroid in self.centroids]
            label = np.argmin(distances)
            labels.append(label)
        return labels
        
        
    def update_centroids(self, data: np.ndarray, labels: List) -> np.ndarray:
        n_features = data.shape[1]
        new_centroids = np.zeros((self.k, n_features))
        counter = [0 for _ in range(self.k)]
        for point, label in zip(data, labels):
            new_centroids[label] += point
            counter[label] += 1
            
        for cluster in range(self.k):
            new_centroids[cluster] /= counter[cluster]
            
        return new_centroids
    
        