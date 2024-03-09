import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
class customlogisticregression(BaseEstimator, ClassifierMixin):
    def __init__(self, learning_rate=0.01, iterations=1000, regularization_strength=0.01):
        self.learning_rate = learning_rate
        self.iterations = iterations
        self.regularization_strength = regularization_strength
        self.weights = None
        self.bias = None
    
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))
    
    def binary_cross_entropy_loss(self, y_true, y_pred):
        # Avoid log(0) which is undefined
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        
        # Compute binary cross entropy loss
        loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
        return loss
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros(n_features)
        self.bias = 0
        
        # Gradient Descent
        for _ in range(self.iterations):
            # Compute predictions
            linear_model = np.dot(X, self.weights) + self.bias
            y_predicted = self.sigmoid(linear_model)

            # Compute loss
            loss = self.binary_cross_entropy_loss(y, y_predicted)
            
            # Compute gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y)) + (self.regularization_strength / n_samples) * self.weights
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # Print loss
            if _ % 100 == 0:
                print(f"Iteration {_}, Loss: {loss}")
    
    def predict_proba(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        proba = self.sigmoid(linear_model)
        return proba
    
    def predict(self, X):
        return (self.predict_proba(X)>0.5).astype("int")
    
    def score(self, X, y=None):
        predictions = self.predict(X)
        accuracy = np.sum(predictions == y) / len(y)
        return accuracy