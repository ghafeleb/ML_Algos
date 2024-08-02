import sys, os
sys.path.insert(0, '../')
import time
from typing import List
from random import random

class KMeans:
    def __init__(self, k: int, tol: float = 1e-5) -> None:
        self.k = k
        self.tol = tol
    
    def stopping_criteria(self, pre_centroids: List[List], centroids: List[List]):
        sum_movement = 0
        for x, y in zip(pre_centroids, centroids):
            sum_movement += self.compute_distance(x, y)
        return sum_movement < self.tol
        
    def fit_1(self, data: List[List], ) -> None:

        assert len(data) > 0, print("Data is empty!")
        assert len(data[0]) > 0, print("Data points are empty!")
        assert len(data[0]) <= self.k, print("Too large k. Each data point is a cluster centroid itself. ")
        
        self.centroids = self.initialize_centroids(data)
        counter_loop = 0
        while True:
            pre_centroids = self.centroids
            labels = self.predict(data,)
            self.centroids = self.update_centroids(data, labels)
            
            if self.stopping_criteria(pre_centroids, self.centroids):
                break
            counter_loop += 1
        print(counter_loop)
    
    def fit_2(self, data: List[List], n_iteration: float) -> None:
        assert len(data) > 0, print("Data is empty!")
        assert len(data[0]) > 0, print("Data points are empty!")
        assert len(data[0]) <= self.k, print("Too large k. Each data point is a cluster centroid itself. ")
        
        self.centroids = self.initialize_centroids(data)
        for i in range(n_iteration):
            pre_centroids = self.centroids
            labels = self.predict(data,)
            self.centroids = self.update_centroids(data, labels)
        
        
    def predict(self, data: List[List], ) -> List:
        labels = []
        for data_point in data:
            min_dist = float('inf')
            min_idx = -1
            for idx, centroid in enumerate(self.centroids):
                dist_temp = self.compute_distance(data_point, centroid)
                if dist_temp < min_dist:
                    min_dist = dist_temp
                    min_idx = idx
            labels.append(min_idx)
                    
        return labels
        
    def get_random_val(self, min_val: float, max_val: float):
        return random() * (max_val - min_val) + min_val
            
    def initialize_centroids(self, data: List[List]) -> List[List]:
        # Find range
        n_data = len(data)
        n_feature = len(data[0])
        feature_min = [float('inf') for i in range(n_feature)]
        feature_max = [-float('inf') for i in range(n_feature)]
        for data_point in data:
            for idx, val in enumerate(data_point):
                feature_min[idx] = min(feature_min[idx], val)
                feature_max[idx] = max(feature_max[idx], val)
            
        centroids = []
        for idx_centroid in range(self.k):
            np.random.seed(42)
            centroid = [self.get_random_val(feature_min[idx], feature_max[idx]) for idx in range(n_feature)]
            centroids.append(centroid)
        return centroids
    
   
    def compute_distance(self, x: List, y: List) -> float:
        dist = 0
        for x_point, y_point in zip(x, y):
            dist += (x_point - y_point)**2
        return dist**0.5
        
    def update_centroids(self, data: List[List], labels: List):
        new_centroids = [[0 for i in range(len(data[0]))] for j in range(self.k)]
        counts = [0 for i in range(self.k)]
        for data_point, label in zip(data, labels):
            counts[label] += 1
            for feature_idx in range(len(data[0])):
                new_centroids[label][feature_idx] += data_point[feature_idx]
        
        for centroid_idx in range(len(new_centroids)):
            if counts[centroid_idx] > 0:
                for feature_idx in range(len(new_centroids[0])):
                    new_centroids[centroid_idx][feature_idx] /= counts[centroid_idx]
                
        return new_centroids
    
        