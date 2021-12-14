import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from hopkins_statistic import hopkins
import numpy as np


def _sse_vs_clusters_plot(x, y, name):
    """
    Plots the SSE vs K, the number of clusters.
    :param x: The number of clusters.
    :param y: The SSE.
    :param name: The filename for the image.
    :return: None, saves a png of the plot.
    """

    plt.title(name)
    plt.plot(x, y)
    plt.xlabel("Cluster Count")
    plt.ylabel("SSE")
    plt.savefig(fname=name)
    plt.show()



def optimal_clusters(X, max_clusters):
    if isinstance(X, pd.DataFrame):
        X = X.values
    sse_list = []
    clusters_list = []
    for cur_clusters in range(1, max_clusters+1):
        kmeans = KMeans(n_clusters=cur_clusters).fit(X)
        sse_list.append(kmeans.inertia_)
        clusters_list.append(cur_clusters)
    _sse_vs_clusters_plot(clusters_list, sse_list, name="SSE vs Clusters")


def main():
    raw_data = pd.read_csv("./ml-1m/ratings.dat", sep="::")
    raw_data.columns = ["user_id", "movie_id", "rating", "timestamp"]
    sorted_by_timestamp = raw_data.sort_values(by=["timestamp"])
    tao = 0.8
    timestamp = sorted_by_timestamp.iloc[round(tao * len(sorted_by_timestamp)), -1]
    data = sorted_by_timestamp[sorted_by_timestamp.timestamp < timestamp]
    unique_users = np.sort(np.unique(data["user_id"]))
    unique_movies = np.sort(np.unique(data["movie_id"]))
    user_mapping = {user: index for index, user in enumerate(unique_users)}
    movie_mapping = {movie: index for index, movie in enumerate(unique_movies)}
    data.loc[:, 'user_id_mapped'] = [user_mapping[user_id] for user_id in data.loc[:, 'user_id']]
    data.loc[:, 'movie_id_mapped'] = [movie_mapping[movie_id] for movie_id in data.loc[:, 'movie_id']]


    X = np.zeros((len(unique_users), len(unique_movies)))
    for i in range(len(unique_users)):
        cur_users_ratings = pd.DataFrame(data[data.user_id_mapped == i]).reset_index(drop=True)
        for j in range(len(cur_users_ratings)):
            movie_index = cur_users_ratings.loc[j, "movie_id_mapped"]
            rating = cur_users_ratings.iloc[j, 2]
            X[i][movie_index] = rating

    optimal_clusters(X, 25)
    print(f"Hopkins statistic: {hopkins(X)} ")



if __name__ == "__main__":
    main()
