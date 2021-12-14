from sklearn.neighbors import NearestNeighbors
from random import sample
from numpy.random import uniform
import numpy as np
import pandas as pd


def hopkins(X):
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X)
    n, d = X.shape
    m = int(0.1 * n)
    nbrs = NearestNeighbors(n_neighbors=1).fit(X.values)

    rand_X = sample(range(0, n, 1), m)

    u_distances = []
    w_distances = []
    for j in range(0, m):
        u_dist, _ = nbrs.kneighbors(uniform(np.amin(X, axis=0), np.amax(X, axis=0), d).reshape(1, -1), 2)
        u_distances.append(u_dist[0][1])
        w_dist, _ = nbrs.kneighbors(X.iloc[rand_X[j]].values.reshape(1, -1), 2, return_distance=True)
        w_distances.append(w_dist[0][1])

    return np.sum(u_distances) / np.sum(np.sum(u_distances) + np.sum(w_distances))
