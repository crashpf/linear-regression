import numpy as np

class LinearRegression:
    '''Linear regression from scratch using gradient descent.
    We scaled "simple_linear_regression.py" to multiple features using vectorization.
    The model now becomes y = Xw + b, where X is a matrix and w is a weight vector.
    '''
    def __init__(self, lr, iterations):
        self.lr = lr
        self.iterations = iterations
        self.w = None
        self.b = None
        self.loss_logs = []
    
    def predict(self, X):
        # X.shape = (n_samples, n_features) 
        return X @ self.w + self.b
    
    def loss_function(self, y, y_pred):
        n = len(y)
        return (1/n) * np.sum((y - y_pred) ** 2)
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.w = np.zeros(n_features)
        self.b = 0

        self.loss_logs = [] # Track losses

        for i in range(self.iterations):
            # y predictions from predict function
            y_pred = self.predict(X)

            # gradients
            dw = (-2 / n_samples) * X.T @ (y - y_pred)  # shape: (n_features,)
            db = (-2 / n_samples) * np.sum(y - y_pred)

            # update weights
            self.w -= self.lr * dw
            self.b -= self.lr * db

            loss = self.loss_function(y, y_pred) # calculate loss
            self.loss_logs.append(loss) # append losses to vector

            # compute loss and track loss
            if i % 1000 == 0:
                print(f"Iteration {i}: Loss = {loss}")

    def get_loss_history(self):
        return self.loss_logs
    
    def score(self, X, y):
        y_pred = self.predict(X)

        ssr = np.sum((y - y_pred)**2)
        ssto = np.sum((y - np.mean(y))**2)
        return 1 - (ssr/ssto) 
    
    def get_coeff(self):
        return {"weights": self.w, "bias": self.b}