import numpy as np
from sklearn.neighbors import NearestNeighbors
from nearest_neighbors import KNNClassifier


def knn_cross_val_score(X, y, k_list, score, cv=None, **kwargs):
    if cv is None:
        cv = kfold(X.shape[0], 3)
    d = {}
    c = 0
    for i in k_list:
        d[i] = np.zeros(len(cv))
    mx = max(k_list)
    for j in cv:
        C = KNNClassifier(k=mx, **kwargs)
        C.fit(X[j[0]], y[j[0]])
        Cl = np.unique(C.y)[np.newaxis, :]
        Cl = np.append(Cl, np.zeros((1, Cl.shape[1])), axis=0)
        if C.weights:
            Yn = C.find_kneighbors(X[j[1]], True)
            for t in range(Yn[1].shape[0]):
                for p in range(mx):
                    Yn[1][t, p] = np.where(Cl[0] == C.y[Yn[1][t, p]])[0]
        else:
            Yn = C.find_kneighbors(X[j[1]], False)
        for i in k_list:
            Ans = np.zeros((X[j[1]].shape[0]))
            if C.weights:
                Yt = (Yn[0][:, :i], Yn[1][:, :i])
                for t in range(Yt[1].shape[0]):
                    Cl[1] = np.zeros(Cl.shape[1])
                    for p in range(i):
                        Cl[1][Yt[1][t, p]] += 1/(Yt[0][t, p] + 10**(-5))
                    Ans[t] = Cl[0, np.argmax(Cl[1])]
            else:
                Yt = Yn[:, :i]
                for t in range(Yn.shape[0]):
                    cnts = np.bincount(C.y[Yt[t, :]])
                    Ans[t] = np.argmax(cnts)
            d[i][c] = np.sum(Ans == y[j[1]])/len(y[j[1]])
        c += 1
    return d


def kfold(n, n_folds):
    z = n % n_folds
    s = 1
    k = n//n_folds
    k1 = 0
    r = []
    for x in range(n_folds):
        if z <= 0:
            s = 0
        l1 = list(range(k1, k1 + s + k))
        l2 = list(range(0, k1))
        l2 = l2 + (list(range(k1 + s + k, n)))
        r.append((np.array(l2), np.array(l1)))
        z -= 1
        k1 += s + k
    return r
