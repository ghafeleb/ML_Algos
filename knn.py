import sys, os
sys.path.insert(0, '../')
import numpy as np
from typing import List
from heapq import heapify, heappush, heappop

"""
1. Compute the distance to all the datapoints (assumption: distance is defined)
2. Select the k closest points (i.e., points with smallest distance) as the closest neighbors
3. Assign the majority vote

"""

def euclidean_dist(x: List, y: List) -> float:
    dist = 0
    for val1, val2 in zip(x, y):
        dist += (val1 - val2) ** 2
    return dist ** 0.5


def knn(k: int, X_train: List[List], y_train: List, X_new: List):
    max_k_distances = []
    for idx, point in enumerate(X_train):
        distance = euclidean_dist(point, X_new)
        if len(max_k_distances) < k:
            heappush(max_k_distances, (-distance, idx))
        else:
            dist_max, idx_max = heappop(max_k_distances)
            if -dist_max > distance:
                heappush(max_k_distances, (-distance, idx))
            else:
                heappush(max_k_distances, (dist_max, idx_max))
    
    closest_neighbors = list(max_k_distances)
    
    counter = [0, 0]
    while max_k_distances:
        _, idx_neighbor = heappop(max_k_distances)
        label = y_train[idx_neighbor]
        counter[label] += 1 
        
    if counter[0] > counter[1]:
        return 0
    else:
        return 1
        
    
        
    