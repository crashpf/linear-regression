class StandardScaler:
    def __init__(self):
        self.mean = None
        self.sd = None
        
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.std = X.std(axis=0)
    
    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Use fit() first")
        return (X - self.mean) / self.std
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)