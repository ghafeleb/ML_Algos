import sys, os
sys.path.insert(0, '../')
import numpy as np
"""
1. Initialize weights
2. Go over loop
    - compute the prediction
    - compute the gradients
    - update the weights
    - check the stopping criteria

"""

class LogisticRegressionNumPy2:
    def __init__(self, n_iter: int, learning_rate: float) -> None:
        self.n_iter = n_iter
        self.learning_rate = learning_rate
    
    def initialize_weights(self, n_feature: int) -> [np.ndarray, float]:
        W = np.zeros(n_feature)
        b = 0
        return W, b
        
    def get_loss(self, y, pred):
        eps = 1e-8
        loss_label_1 = y * np.log(pred+eps)
        loss_label_0 = (1-y) * np.log(1 - pred+eps)
        bce = -(loss_label_1 + loss_label_0).mean()
        return bce
    
    def get_gradients(self, X, pred_prob, y):
        dW = X.T.dot(pred_prob - y) / X.shape[0]
        db = (pred_prob - y).mean()
        return dW, db
        
    def update_weights(self, dW, db):
        self.W -= self.learning_rate * dW
        self.b -= self.learning_rate * db
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        self.loss_values = []
        # Initialize the weights
        self.W, self.b = self.initialize_weights(X.shape[1])
        
        # Compute loss
        pred_prob = self.predict_prob(X,)
        pre_loss = self.get_loss(y, pred_prob)
        self.loss_values.append(pre_loss)
        
        # Loop
        for iter in range(self.n_iter):
            # Get the prediction
            pred_prob = self.predict_prob(X)
            
            # Update the weights using GD
            dW, db = self.get_gradients(X, pred_prob, y)
            self.update_weights(dW, db)
            
            # Compute loss for iteration 
            pred_prob = self.predict_prob(X,)
            loss = self.get_loss(y, pred_prob)
            self.loss_values.append(pre_loss)
            
#             if pre_loss < loss:
#                 break
                
#             pre_loss = loss
        
    def sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
    
    def predict_prob(self, X: np.ndarray) -> np.ndarray:
        linear_func_val = X.dot(self.W) + self.b
        probs = self.sigmoid(linear_func_val)
        return probs
    
    def predict_labels(self, X_test, threshold):
        pred_prob = self.predict_prob(X_test)
        labels = [1 if pred_prob[idx] > threshold else 0 for idx in range(len(pred_prob))]
        return labels

def accuracy(pred, y):
    correctly_classified = [pred[i] == y[i] for i in range(len(pred))]
    return sum(correctly_classified)/len(y)

class LogisticRegressionNumPy:
    def __init__(self, n_iterations: int = 100, learning_rate: float = 0.01) -> None:
        self.n_iterations = n_iterations
        self.learning_rate = learning_rate
        self.counter = 10
    
    def initialize_weights(self, X):
        np.random.seed(42)
        n_features = X.shape[1]
        w = np.zeros(n_features)
        b = 0
        return w, b
    
    #Sigmoid method
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict_prob(self, X): # predict probabilities
        z = np.dot(X, self.w) + self.b
        return self._sigmoid(z)
        
    def predict_labels(self, X, threshold):
        pred = self.predict_prob(X)
        labels = [1 if pred[idx]>threshold else 0 for idx in range(len(pred))]
        return labels
        
    def get_loss(self, pred, y):
        # binary cross entropy or log loss
        epsilon = 1e-9
        log_loss = -(y * np.log(pred + epsilon) + (1 - y) * np.log(1 - pred + epsilon)).mean()
        return log_loss
        
    def get_gradients(self, X, y, pred):
        dw = np.dot(X.T, pred - y) / X.shape[0]
        db = np.sum(pred - y) / X.shape[0]
#         print(dw, db)
        return dw, db
    
    def update_params(self, dw, db):
        self.w -= self.learning_rate * dw
        self.b -= self.learning_rate * db
        
    def fit(self, X, y, bias = False):
        # Initialize the weights
        self.w, self.b = self.initialize_weights(X)
        
        # Compute the loss
        pred = self.predict_prob(X) # predict
        pre_loss = self.get_loss(pred, y) # compute the loss
        
        for iter in range(self.n_iterations):
#             print(self.w, self.b)
            # Compute the prediction
            pred_prob = self.predict_prob(X) # predict
            # Compute the gradients
            dw, db = self.get_gradients(X, y, pred_prob)
            
            # Update the parameters
            self.update_params(dw, db)

            # Re-compute the loss
            pred_prob = self.predict_prob(X) # predict
            loss = self.get_loss(pred_prob, y) # compute the loss
            
#             # Check the stopping criteria: immediately stop when loss increase from last iteration
#             if loss  >=  pre_loss:
#                 self.counter -= 1
#                 if self.counter == 0:
#                     break
            
            # Update the pre_loss
            pre_loss = loss
        
        
# https://medium.com/@koushikkushal95/logistic-regression-from-scratch-dfb8527a4226  
class LogisticRegressionBaseline:
    def __init__(self, learning_rate=0.001, n_iters=1000):
        self.lr = learning_rate
        self.n_iters = n_iters
        self.weights = None
        self.bias = None
        self.losses = []
         
    #Sigmoid method
    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def compute_loss(self, y_true, y_pred):
        # binary cross entropy
        epsilon = 1e-9
        y1 = y_true * np.log(y_pred + epsilon)
        y2 = (1-y_true) * np.log(1 - y_pred + epsilon)
        return -np.mean(y1 + y2)

    def feed_forward(self,X):
        z = np.dot(X, self.weights) + self.bias
        A = self._sigmoid(z)
        return A

    def fit(self, X, y):
        n_samples, n_features = X.shape

        # init parameters
        self.weights = np.zeros(n_features)
        self.bias = 0

        # gradient descent
        for _ in range(self.n_iters):
#             print(self.weights, self.bias)
            A = self.feed_forward(X)
#             print(A[:5])
            self.losses.append(self.compute_loss(y,A))
            dz = A - y # derivative of sigmoid and bce X.T*(A-y)
            # compute gradients
            dw = (1 / n_samples) * np.dot(X.T, dz)
            db = (1 / n_samples) * np.sum(dz)
#             print(dw, db)
            # update parameters
            self.weights -= self.lr * dw
            self.bias -= self.lr * db
            
    def predict(self, X):
        threshold = .5
        y_hat = np.dot(X, self.weights) + self.bias
        y_predicted = self._sigmoid(y_hat)
        y_predicted_cls = [int(1) if i > threshold else int(0) for i in y_predicted]
        
        return y_predicted_cls
