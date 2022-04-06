import numpy as np
from numpy import power, sum, abs, count_nonzero


class KNearestNeighborsClassifier:
    def __init__(self, n_neighbors=3, metric='minkowski', p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X = X
        self.y = y

        return self

    def _get_manhattan_distance(self, diff):
        self.p = 1
        return self._get_minkowski_distance(diff)

    def _get_euclidean_distance(self, diff):
        self.p = 2
        return self._get_minkowski_distance(diff)

    def _get_minkowski_distance(self, diff):
        return power(sum(power(abs(diff), self.p), axis=1), (1 / self.p))

    def _get_distances(self, X):
        diff = self.X - X
        if self.metric == "euclidean":
            distance = self._get_euclidean_distance(diff)
        elif self.metric == "minkowski":
            distance = self._get_minkowski_distance(diff)
        elif self.metric == "manhattan":
            distance = self._get_manhattan_distance(diff)
        else:
            raise NotImplementedError("Please, use 'euclidean' or 'minkowski' distance")
        return distance

    def _predict_one_shot(self, X):
        distances = self._get_distances(X)

        sorted_dist_arg = distances.argsort()[:self.n_neighbors]
        labels_vote = self.y[sorted_dist_arg]
        vote_dict = {key: count_nonzero(labels_vote == key) for key in np.unique(self.y)}
        vote_dict_sorted = sorted(vote_dict.items(), key=lambda item: item[1], reverse=True)

        return vote_dict_sorted[0][0]

    def _predict_proba_one_shot(self, X):
        distances = self._get_distances(X)

        sorted_dist_arg = distances.argsort()[:self.n_neighbors]
        labels_vote = self.y[sorted_dist_arg]
        return [count_nonzero(labels_vote == key) / len(labels_vote) for key in np.unique(self.y)]

    def predict(self, X):
        return np.array([self._predict_one_shot(x) for x in X])

    def predict_proba(self, X):
        return np.array([self._predict_proba_one_shot(x) for x in X])
