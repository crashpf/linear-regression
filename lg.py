import numpy as np

class LinearRegression():

    def __init__(self, lr, iteration):
        self.lr = lr
        self.iteration = iteration
        self.w = None
        self.b = None
        self.loss_logs = []

    def predict(self, X):
        return X @ self.w + self.b
    
    def loss_function(self, y, y_pred):
        n = len(y)

        return 1/n * np.sum((y - y_pred)**2)

    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0
        self.loss_logs = []

        for i in range(self.iteration):
            y_pred = self.predict(X)

            # gradients
            dw = (-2 / n_samples) * X.T @ (y - y_pred)  # shape: (n_features,)
            db = (-2 / n_samples) * np.sum(y - y_pred)

            

    