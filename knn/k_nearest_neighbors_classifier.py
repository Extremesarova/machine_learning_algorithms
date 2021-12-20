import numpy as np
from numpy import power, sum, abs, sqrt, count_nonzero


class KNearestNeighborsClassifier:
    def __init__(self, n_neighbors=3, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y

    def _predict_one_example(self, X):
        distance = 0
        diff = self.X - X
        if self.metric == 'euclidean':
            distance = get_euclidean_distance(diff)
        elif self.metric == 'minkowski':
            distance = get_minkowski_distance(diff, self.p)
        sorted_dist_arg = distance.argsort()[:self.n_neighbors]
        labels_vote = self.y[sorted_dist_arg]
        vote_dict = {key: count_nonzero(labels_vote == key) for key in np.unique(self.y)}
        vote_dict_sorted = sorted(vote_dict.items(), key=lambda item: item[1], reverse=True)
        return vote_dict_sorted[0][0]

    def predict(self, X):
        out = []
        for x in X:
            out.append(self._predict_one_example(x))
        return np.array(out)


def get_euclidean_distance(diff):
    return sqrt(sum(power(diff, 2), axis=1))


def get_minkowski_distance(diff, p):
    return power(sum(power(abs(diff), p), axis=1), (1 / p))
