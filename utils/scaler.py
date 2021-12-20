import numpy as np


class StandardScaler:
    def __init__(self, is_fit=False):
        self.is_fit = is_fit

    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.is_fit = True

    def transform(self, X):
        if self.is_fit:
            return (X - self.mean) / self.std
        else:
            print("You should perform fit before calling transform")

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)