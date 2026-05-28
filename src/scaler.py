class StandardScaler:
    """
    From scratch implementation of StandardScaler by sklearn.
    Standardizes features using:
        z = (x - mean) / std
    """
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
    
    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Use fit() first")
        return (X - self.mean) / self.sd
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)