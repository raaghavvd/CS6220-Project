""""
import data, create train and test matrix
creating rating matrix
create a shared nearest neighbors graph
cluster the data by running DBSCAN on the SNN graph
assign test data points to the appropriate cluster
predict the rating for a movie a based on the average rating of the cluster
calculate the MAE of the predictions
"""

from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error




def get_movie_genres():
    db = pd.read_csv(".\\ml-1m\\movies.dat", delimiter="::", header=None, encoding='latin-1')
    genres = db.loc[:, 2]
    movie_ids = db.loc[:, 0]
    genres_per_movie_dict = {}
    unique_genres_dict = {}
    index = 0
    for movie_genre_string, movie_id in zip(genres, movie_ids):
        movie_genres = movie_genre_string.split("|")
        for genre in movie_genres:
            if genre not in unique_genres_dict:
                unique_genres_dict[genre] = index
                index += 1
        genres_per_movie_dict[movie_id] = movie_genres

    return genres_per_movie_dict, unique_genres_dict




def get_ratings_matrix(data, unique_users, unique_movies, user_mapping, movie_mapping):
    X = np.zeros((len(unique_users), len(unique_movies)))
    for user in unique_users:
        user_ratings = pd.DataFrame(data[data.user_id == user]).reset_index(drop=True)
        user_index = user_mapping[user]
        for j in range(len(user_ratings)):
            movie_index = movie_mapping[user_ratings.loc[j, "movie_id"]]
            rating = user_ratings.loc[j, "rating"]
            X[user_index][movie_index] = rating
    return X


def get_user_genres(data, users, movie_genre_dict, genre_map):
    user_genres = np.zeros((len(users), len(genre_map.keys())))
    for i in range(len(users)):
        user_data = data[data.user_id == users[i]]
        for j in range(len(user_data)):
            movie_id = user_data.iloc[j, 1]
            movie_genres = movie_genre_dict[movie_id]
            for k in range(len(movie_genres)):
                movie_index = genre_map[movie_genres[k]]
                user_genres[i][movie_index] += 1
    scaler = StandardScaler()
    return scaler.fit_transform(user_genres)




def predict_ratings(prediction_indexes, clusters_dict, ratings, user_cluster_assignment):
    prediction_values = []
    average_ratings = np.nan_to_num(np.sum(ratings, axis=0) / np.count_nonzero(ratings, axis=0), 1)
    average_ratings[np.where(average_ratings == 0)] = 3
    cluster_averages = []
    for key in clusters_dict.keys():
        cluster = clusters_dict[key]
        cluster_predictions = np.nan_to_num(np.sum(ratings[cluster - 1], axis=0) / np.count_nonzero(ratings[cluster - 1], axis=0), 1)
        cluster_averages.append(cluster_predictions)
    row_indexes = prediction_indexes[0]
    col_indexes = prediction_indexes[1]
    for user_index, movie_index in zip(row_indexes, col_indexes):
        user_id = user_index + 1
        if user_id in user_cluster_assignment:
            user_cluster = user_cluster_assignment[user_id]
            if -1 in clusters_dict:
                user_cluster = user_cluster + 1
            predicted_value = cluster_averages[user_cluster][movie_index]
            if not predicted_value:
                # cluster hasn't seen the movie
                prediction_values.append(average_ratings[movie_index])
            else:
                # have a valid prediction
                prediction_values.append(predicted_value)
        else:
            # No user information so default to the average
            prediction_values.append(average_ratings[movie_index])

        if min(prediction_values) == 0:
            print("inserted 0")
    return prediction_values






def main():
    raw_data = pd.read_csv("./ml-1m/ratings.dat", sep="::")
    raw_data.columns = ["user_id", "movie_id", "rating", "timestamp"]

    unique_users = np.sort(np.unique(raw_data["user_id"]))
    unique_movies = np.sort(np.unique(raw_data["movie_id"]))

    sorted_by_timestamp = raw_data.sort_values(by=["timestamp"])
    tao = 0.8
    timestamp = sorted_by_timestamp.iloc[round(tao * len(sorted_by_timestamp)), -1]
    train_data = sorted_by_timestamp[sorted_by_timestamp.timestamp < timestamp]
    test_data = sorted_by_timestamp[sorted_by_timestamp.timestamp >= timestamp]

    train_users = np.sort(np.unique(train_data["user_id"]))
    # test_users = np.sort(np.unique(test_data[["user_id"]]))

    movie_genre_dict, genre_map = get_movie_genres()

    user_mapping = {user: index for index, user in enumerate(unique_users)}
    movie_mapping = {movie: index for index, movie in enumerate(unique_movies)}

    X_train = get_user_genres(train_data, train_users, movie_genre_dict, genre_map)
    # X_test = get_user_genres(test_data, test_users, movie_genre_dict, genre_map)

    ratings_train = get_ratings_matrix(train_data, unique_users, unique_movies, user_mapping, movie_mapping)
    ratings_test = get_ratings_matrix(test_data, unique_users, unique_movies, user_mapping, movie_mapping)

    eps_list = [0.1, 0.3, 0.5, 0.75]
    min_point_list = [3, 5, 5, 10]
    metric_list = ["euclidean", "cosine", "minkowski"]
    for metric in metric_list:
        for eps, min_points in zip(eps_list, min_point_list):

            distances = pairwise_distances(X_train, metric=metric)
            db = DBSCAN(eps=eps, min_samples=min_points, metric="precomputed")
            db.fit_predict(distances)
            cluster_dict = {}
            user_cluster_assignment = {}
            clusters, counts = np.unique(db.labels_, return_counts=True)
            for label in clusters:
                cluster_dict[label] = train_users[db.labels_ == label]
                for user in train_users[db.labels_ == label]:
                    user_cluster_assignment[user] = label


            test_list = np.where(ratings_test > 0)
            prediction_list = np.round(predict_ratings(test_list, cluster_dict, ratings_train, user_cluster_assignment))
            true_vals_list = ratings_test[np.where(ratings_test > 0)]


            MAE = mean_absolute_error(true_vals_list, prediction_list)


            with open(f"cluster_results_{metric}_eps_{eps}", "w") as _file:
                _file.write(f"DBSCAN Params:\n"
                            f"    Metric: {metric}\n"
                            f"    EPS: {eps}\n"
                            f"    Min Points:{min_points}\n"
                            f"    Number of Clusters: {clusters}\n"
                            f"    Cluster Sizes: {counts}\n"
                            f"    MAE: {MAE}\n ")
                _file.close()






if __name__ == "__main__":
    main()
