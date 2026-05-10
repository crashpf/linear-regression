import numpy as np

class LinearRegression:
    '''Linear regression from scratch using gradient descent.
    We scaled "simple_linear_regression.py" to multiple features using vectorization.
    The model now becomes y = Xw + b, where X is a matrix and w is a weight vector.
    '''
    def __init__(self, lr, iter):
        self.lr = lr
        self.iter = iter
        self.w = None
        self.b = 0
    
    def predict(self, x):
        # X.shape = (n_samples, n_features) 
        return