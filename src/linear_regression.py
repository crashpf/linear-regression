import numpy as np

class LinearRegression:
    """
    Linear Regression from scratch using gradient descent.
    Supports multivariate regression:
        y = Xw + b
    """
    def __init__(self, lr: float, iterations: int) -> None:
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = None
        self.loss_logs = []
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        # X.shape = (n_samples, n_features) 
        return X @ self.w + self.b
    
    def loss_function(self, y: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Mean squared error loss function
        """
        n = len(y)
        return (1/n) * np.sum((y - y_pred) ** 2)
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Model training using gradient descent to update the weights.
        w = w - lr*dw
        b = b - lr*db
        """
        n_samples, n_features = X.shape
        
        self.w = np.zeros(n_features)
        self.b = 0

        self.loss_logs = [] # Track losses

        for i in range(self.iterations):
            # y predictions from predict function
            y_pred = self.predict(X)

            loss = self.loss_function(y, y_pred) # calculate loss
            self.loss_logs.append(loss) # append losses to vector

            # gradients
            dw = (-2 / n_samples) * X.T @ (y - y_pred)  # shape: (n_features,)
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

            # compute loss and track loss
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss = {loss}")
    
    def score(self, X: np.ndarray, y: np.ndarray) -> float:
        y_pred = self.predict(X)

        ssr = np.sum((y - y_pred)**2)
        ssto = np.sum((y - np.mean(y))**2)
        return 1 - (ssr/ssto) 
