import numpy as np

class StandardScaler:
    """
    Implementation of StandardScaler by sklearn from scratch.
    Standardizes features X using:
        z = (x - mean) / std
    """
    def __init__(self) -> None:
        self.mean = None
        self.std = None
        
    def fit(self, X: np.ndarray) -> None:
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.mean is None:
            raise RuntimeError("Use fit() first")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self.transform(X)
    
    def inverse_transform(self, X_scaled: np.ndarray) -> np.ndarray:
        return (X_scaled * self.std) + self.mean