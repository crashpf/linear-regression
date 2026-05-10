import numpy as np

class SimpleLinearRegression:
    '''Simple linear regression using gradient descent from scratch for a single feature X1 (y = mx + b). 
    The goal is to optimize the parameters m and b which are scalars.
    '''
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        self.m = 0
        self.b = 0

    def predict(self, x):
        # x.shape = (n_samples, n_features), in this case (200,1)
        return self.m * x + self.b 

    def loss_function(self, y, y_pred):
        n = len(y)
        return (1/n) * np.sum((y - y_pred)**2)
        
    def fit(self, x, y):
        n = len(x)

        for i in range(self.iterations):
            # get y_predictions from predict function
            y_pred = self.predict(x) 

            # compute gradients 
            dm = (-2/n) * np.sum(x * (y - y_pred))
            db = (-2/n) * np.sum(y - y_pred)
            
            # update parameters
            self.m = self.m - self.lr * dm
            self.b = self.b - self.lr * db

            # compute loss and track loss
            if i % 1000 == 0:
                loss = self.loss_function(y, y_pred)
                print(f"Iteration {i}: Loss = {loss}, m = {self.m}, b = {self.b}")
