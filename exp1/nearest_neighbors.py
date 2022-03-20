import numpy as np
from sklearn.neighbors import NearestNeighbors
from distances import euclidean_distance, cosine_distance


class KNNClassifier:

    def __init__(self, k, strategy, metric, weights, test_block_size):
        self.k = k
        self.s = strategy
        self.m = metric
        self.w = weights
        self.tbs = test_block_size

    def fit(self, X, y):
        if self.s == 'my_own':
            self.X = X
        elif self.s == 'ball_tree':
            self.temp = NearestNeighbors(n_neighbors=self.k, algorithm=self.s, metric='euclidean')
            self.temp.fit(X)
        elif self.s == 'kd_tree':
            self.temp = NearestNeighbors(n_neighbors=self.k, algorithm=self.s, metric='euclidean')
            self.temp.fit(X)
        elif self.s == 'brute':
            self.temp = NearestNeighbors(n_neighbors=self.k, algorithm=self.s, metric=self.m)
            self.temp.fit(X)
        self.y = y

    def find_kneighbors(self, X, return_distance):
        if self.s == 'my_own':
            Z = np.zeros((X.shape[0], self.k))
            if self.m == 'cosine':
                M = cosine_distance(X, self.X)
            else:
                M = euclidean_distance(X, self.X)
            ind = M.argsort(axis=1)[:, :self.k]
            if return_distance:
                for i in range(X.shape[0]):
                    for j in range(self.k):
                        Z[i, j] = M[i, ind[i, j]]
                return((Z, ind))
            else:
                return ind
        elif self.s == 'ball_tree':
            return self.temp.kneighbors(X=X, n_neighbors=self.k, return_distance=return_distance)
        elif self.s == 'kd_tree':
            return self.temp.kneighbors(X=X, n_neighbors=self.k, return_distance=return_distance)
        elif self.s == 'brute':
            return self.temp.kneighbors(X=X, n_neighbors=self.k, return_distance=return_distance)

    def predict(self, X):
        Ans = np.zeros((X.shape[0]))
        Cl = np.unique(self.y)[np.newaxis, :]
        Cl = np.append(Cl, np.zeros((1, Cl.shape[1])), axis=0)
        if self.w:
            Y = self.find_kneighbors(X, True)
            for i in range(Y[1].shape[0]):
                Cl[1] = np.zeros(Cl.shape[1])
                for j in range(self.k):
                    for k in range(Cl.shape[1]):
                        if self.y[Y[1][i, j]] == Cl[0, k]:
                            Cl[1, k] += 1/(Y[0][i, j] + 10**(-5))
                Ans[i] = Cl[0, np.argmax(Cl[1])]
        else:
            Y = self.find_kneighbors(X, False)
            for i in range(Y.shape[0]):
                Cl[1] = np.zeros(Cl.shape[1])
                for j in range(self.k):
                    for k in range(Cl.shape[1]):
                        if self.y[Y[i, j]] == Cl[0, k]:
                            Cl[1, k] += 1
                Ans[i] = Cl[0, np.argmax(Cl[1])]
        return Ans
