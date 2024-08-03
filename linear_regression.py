import sys, os
sys.path.insert(0, '../')
import numpy as np
import math


class LinearRegression:
    def __init__(self, n_iter : int, learning_rate: float) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
        self.losses = []
        
    def predict(self, X):
        predictions = []
        for x in X:
            prediction = self.b
            for w, feature_val in zip(self.W, x):
                prediction += feature_val * w
            predictions.append(prediction)
        return predictions
            
       
    def initialize_weights(self, n_features):
        self.W = [0 for i in range(n_features)]
        self.b = 0  
            
    # MSE
    def compute_mse(self, predictions, y): 
        loss = 0
        # Sum loss of every single point
        for prediciton, y_val in zip(predictions, y):
            loss += (prediciton - y_val) ** 2
        # Return averaged value
        return loss / len(y)
    
    def get_gradients(self, X, predictions, y):
        # sum_j((sum_i(wi * x_ij) + b - y_label_j) ** 2
        # single data point, derivative wrt w_i: 2 * x_ij * (sum_i(wi * x_ij) + b - y_label_j) / n_sample
        # single data point, derivative wrt w_i: 2 * x_ij * (prediction - y_label_j) / n_sample
        # single data point, derivative wrt b: 2 *(sum_i(wi * x_ij) + b - y_label_j) / n_sample
        # single data point, derivative wrt b: 2 *(prediction - y_label_j) / n_sample
        n_feature = len(X[0])
        n_sample = len(X)
        dW = [0 for i in range(n_feature)]
        db = 0
        for x, prediction, y_val in zip(X, predictions, y):
            db += 2 * (prediction - y_val) / n_sample
            for idx_feature in range(n_feature):
                dW[idx_feature] += 2 * x[idx_feature] * (prediction - y_val) / n_sample
        return dW, db
    
    def update_weights(self, dW, db):
        self.b -= self.learning_rate * db
        for idx_feature, dW_val in enumerate(dW):
            self.W[idx_feature] -= self.learning_rate * dW_val
        
    def fit(self, X, y):
        # Initialize the weights
        n_features = len(X[0])
        self.initialize_weights(n_features)
        
        # Compute the loss
        predictions = self.predict(X)
        loss = self.compute_mse(predictions, y)
        self.losses.append(loss)
        
        # Loop
        for iter in range(self.n_iter):
            # Compute the predictions
            predictions = self.predict(X)
            # Compute the gradients
            dW, db = self.get_gradients(X, predictions, y)
            # Update the weights
            self.update_weights(dW, db)
            
            # Re-compute the loss
            predictions = self.predict(X)
            loss = self.compute_mse(predictions, y)
            self.losses.append(loss)
            
            
    