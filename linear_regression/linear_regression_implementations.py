import copy

import numpy as np
from numpy import matmul, dot
from numpy.dual import pinv
from numpy.linalg import det
from scipy.linalg import lstsq
from scipy.optimize import nnls


class SciPyLinearRegressionOLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = copy.deepcopy(X)
        X.insert(0, "dummy", 1)
        p, _, _, _ = lstsq(X, y)
        self.intercept_ = p[0]
        self.coef_ = p[1:]


class SciPyLinearRegressionNNOLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = copy.deepcopy(X)
        X.insert(0, "dummy", 1)
        p, _, = nnls(X, y)
        self.intercept_ = p[0]
        self.coef_ = p[1:]


class LinearRegressionOLS:
    def __init__(self):
        pass

    def fit(self, X, y):
        X = copy.deepcopy(X)
        X.insert(0, "dummy", 1)
        if det(np.matmul(X.values.T, X.values)) == 0.0:
            print("This matrix is singular, cannot do inverse operation")
            self.intercept_ = None
            self.coef_ = None
        else:
            weights = matmul(pinv(np.matmul(X.values.T, X.values)), dot(X.values.T, y))
            self.intercept_ = weights[0]
            self.coef_ = weights[1:]
