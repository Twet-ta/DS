import numpy as np


def euclidean_distance(X, Y):
    Z = np.sqrt(X**2@np.ones((X.shape[1], 1)) - 2*(X@Y.transpose()) + (Y**2@np.ones((Y.shape[1], 1))).transpose())
    return Z


def cosine_distance(X, Y):
    M = X@Y.transpose()*((np.sqrt(X**2@np.ones((X.shape[1], 1))))**(-1))
    Z = np.ones((X.shape[0], Y.shape[0])) - M*((np.sqrt(Y**2@np.ones((Y.shape[1], 1))).transpose())**(-1))
    return Z
