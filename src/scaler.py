import numpy as np

class StandardScaler:
    """
    From scratch implementation of StandardScaler by sklearn.
    Standardizes features using:
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