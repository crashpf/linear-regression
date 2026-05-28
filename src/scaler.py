class StandardScaler:
    ''' Manual implementation of the StandardScaler from sklearn.'''
    def __init__(self):
        self.mean = None
        self.sd = None
        
    def fit(self, X):
        self.mean = X.mean(axis=0)
        self.sd = X.sd(axis=0)
    
    def transform(self, X):
        if self.mean is None:
            raise RuntimeError("Use fit() first")
        return (X - self.mean) / self.sd
    
    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)