from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import random
import json
from sklearn.metrics import mean_absolute_error


MOVIEID = "movie_id"
RATING = "rating"


def create_genre_dict():
    DELIMINATOR = "|"
    SEPARATOR = ","
    fn = "netflix_movie_genres.txt"
    _file = open(f"C:\\Users\\mmeuse\\Mark Docs\\CS6220 Data Mining\\Project\\archive\\{fn}")
    result = _file.readlines()
    genre_map = {}
    skip_first = True
    i = 0
    for line in result:
        if skip_first:
            skip_first = False
            continue
        genres = line[line.index(SEPARATOR):].split(DELIMINATOR)
        for genre in genres:
            genre = genre.rstrip()
            genre = genre.replace(",", "")
            if genre not in genre_map.values():
                genre_map[i] = genre
                i += 1
    return genre_map


def select_genres(genre_map_keys, samples=4):
    """
    Selects random genres without replacement to avoid duplicates
    :param genre_map_keys: keys of the genre dict
    :param samples: number of samples to draw
    :return: list of ints, the genres
    """
    return random.sample(genre_map_keys, samples)


def create_cluster_matrix(n, d=19, num_genres=4):
    m = np.empty((n, d))
    for i in range(n):
        m[i] = create_user_genre_vector(d, num_genres)
    return m


def create_user_genre_vector(d, num_genres):
    v = np.zeros((d,), dtype="float16")
    genre_index_list = select_genres(range(d), num_genres)
    for index in genre_index_list:
        val = np.float16(random.random())
        v[index] = val
    v = v / np.sum(v)
    return v


def process_data(users, data, genre_matrix, num_genres=19,):
    unique_movies = np.sort(np.unique(data["movie_id"]))
    movie_mapping = {movie: index for index, movie in enumerate(unique_movies)}
    m = np.zeros((len(users), num_genres))
    pv = np.zeros((len(users), 1), dtype=int)
    for u in range(len(users)):
        user_data = data[data.user_id == users[u]]
        for i in range(len(user_data)):
            movie_id = user_data.iloc[i, 1]
            movie_genres = genre_matrix[movie_mapping[movie_id], :]
            for j in range(len(movie_genres)):
                if movie_genres[j] > 0:
                    m[u][j] += 1
        pv[u] = int(np.argmax(m[u]))
        m[u] = m[u] / np.sum(m[u])

    return m, pv




def Voting(num_clusters, cluster_dict):
    """
    Calculates the Borda vote count for each cluster.
    :param num_clusters: (int) number of clusters
    :param cluster_array: Numpy array, the clustered training data of user genre preferences
    :return: Numpy array, of shape (num_clusters, num_genres)
    """
    k = cluster_dict[0].shape[1] # number of dimensions in a cluster
    rating_vector = np.empty((num_clusters, k))
    for i in range(num_clusters):
        rating_vector[i] = CalculateRating(cluster_dict[i])
    return rating_vector



def CalculateRating(cluster):
    """
    Return a vector of length k where k = the number of unique movie genres. The values in the vector represent the
    number of users with a preference for genre[k].
    :param cluster: cluster of similar users produced by DBSCAN
    :return: Numpy array of ints, a sum for each genre (ie column) in the cluster matrix.
    """
    return np.sum(cluster, axis=0)



def calculate_average_movie_ratings():
    db = pd.read_csv(".\\ml-100k\\u.data", delim_whitespace=True)
    average_ratings = db.groupby([MOVIEID]).mean()[RATING]
    return average_ratings



def get_genre_dict():
    db = pd.read_csv(".\\ml-100k\\u.genre", delimiter="|", header=None)
    genre_dict = db.to_dict()[0]
    return genre_dict


def get_movie_genres():
    db = pd.read_csv(".\\ml-100k\\u.item", delimiter="|", header=None, encoding='latin-1')
    genre_map = db.values[:, -19:]
    return genre_map


def get_movie_recommendations(user, top_movies_per_genre):
    user_genre = user.argmax()
    return top_movies_per_genre[user_genre]


def get_top_10_movies_per_genre(genre_dict, movie_genres, average_ratings, movie_titles):
    top10_dict = {}
    assert len(movie_genres[:, 0]) == len(average_ratings)
    for i in range(len(movie_genres[:, 0])):
        all_genres_for_movie = movie_genres[i, :]
        for j in range(len(all_genres_for_movie)):
            if all_genres_for_movie[j] > 0:
                genre_string = genre_dict[j]
                if genre_string not in top10_dict:
                    top10_dict[genre_string] = []
                    top10_dict[genre_string].append((i, average_ratings.values[i]))
                else:
                    top10_dict[genre_string].append((i, average_ratings.values[i]))

    for key in top10_dict.keys():
        top10 = top10_dict[key]
        top10.sort(key=lambda x: x[1], reverse=True)
        if len(top10) > 10:
            top10 = top10[:10]
        top10_movie_names = []
        for movie_tuple in top10:
            movie_index = movie_tuple[0]
            movie_name = movie_titles[movie_index]
            top10_movie_names.append(movie_name)
        top10_dict[key] = top10_movie_names

    return top10_dict


def predict_ratings(train_data, ratings, training_genre_preferences, movie_genre_matrix):
    prediction_matrix = np.zeros(train_data.shape)
    for i in range(train_data.shape[0]):
        preferred_genre = training_genre_preferences[i]
        for j in range(train_data.shape[1]):
            if train_data[i][j] < 1: # user hasn't seen the movie, so make a prediction
                if movie_genre_matrix[j][preferred_genre] > 1: # is in the user's preferred genre
                    prediction_matrix[i][j] = ratings[j + 1]
                else:
                    prediction_matrix[i][j] = ratings[j + 1]

    return prediction_matrix



def make_recommendations(user_matrix, cluster_votes, genre_dict, top_movies_per_genre):
    n, k = user_matrix.shape
    recommendation_dict = {}
    for i in range(n):
        cluster_assignment = assign_user_to_cluster(user_matrix[i], cluster_votes)
        preferred_genre_of_cluster = np.argmax(cluster_votes[cluster_assignment])
        recommendation_dict[i] = top_movies_per_genre[genre_dict[preferred_genre_of_cluster]]
    return recommendation_dict


def write_dict_to_file(filename, out_dict):
    if not filename.lower().endswith(".json"):
        filename = filename + ".json"
    with open(filename, "w") as _file:
        _file.write(json.dumps(out_dict))
    _file.close()


def assign_user_to_cluster(user, cluster_votes):
    user_preferred_genre = np.argmax(user, axis=0)
    vote_list = []
    for cluster in cluster_votes:
        vote_list.append(cluster[user_preferred_genre])
    return np.argmax(vote_list)


def get_movie_title_dict():
    db = pd.read_csv(".\\ml-100k\\u.item", delimiter="|", header=None, encoding='latin-1')
    db = db.iloc[:, 1:2]
    return db.to_dict()[1]


def preferred_movies(genre_preferences, top_10_movies_per_genre, genre_dict):
    recommendations = np.empty((len(genre_preferences), 10), dtype=int)
    gp = genre_preferences.squeeze()
    for i in range(len(genre_preferences)):
        recommendations[i] = top_10_movies_per_genre[genre_dict[gp[i]]]
    return recommendations


def get_ratings_matrix(data):
    unique_users = np.sort(np.unique(data["user_id"]))
    unique_movies = np.sort(np.unique(data["movie_id"]))
    user_mapping = {user: index for index, user in enumerate(unique_users)}
    movie_mapping = {movie: index for index, movie in enumerate(unique_movies)}
    data.loc[:, 'user_id_mapped'] = [user_mapping[user_id] for user_id in data['user_id']]
    data.loc[:, 'movie_id_mapped'] = [movie_mapping[movie_id] for movie_id in data['movie_id']]

    X = np.zeros((len(unique_users), len(unique_movies)))
    for i in range(len(unique_users)):
        cur_users_ratings = pd.DataFrame(data[data.user_id_mapped == i]).reset_index(drop=True)
        for j in range(len(cur_users_ratings)):
            movie_index = cur_users_ratings.loc[j, "movie_id_mapped"]
            rating = cur_users_ratings.iloc[j, 2]
            X[i][movie_index] = rating
    return X




def main():
    random.seed(1)
    # number of users
    n = 60000
    # number of genres
    d = 19
    cluster_matrix = create_cluster_matrix(n, d)
    raw_data = pd.read_csv("ml-100k/u.data", sep="\t")
    sorted_by_timestamp = raw_data.sort_values(by=["timestamp"])
    tao = 0.8
    timestamp = sorted_by_timestamp.iloc[round(tao * len(sorted_by_timestamp)), -1]
    training_data = sorted_by_timestamp[sorted_by_timestamp.timestamp < timestamp].sort_values(by=["user_id", "movie_id"])
    test_data = sorted_by_timestamp[sorted_by_timestamp.timestamp >= timestamp].sort_values(by=["user_id", "movie_id"])
    train_users = np.sort(pd.unique(training_data.get("user_id")))
    test_users = np.sort(pd.unique(test_data.get("user_id")))
    movie_title_dict = get_movie_title_dict()
    genre_dict = get_genre_dict()
    movie_genre_matrix = get_movie_genres()
    average_ratings = calculate_average_movie_ratings()
    top_10_movies_per_genre = get_top_10_movies_per_genre(genre_dict, movie_genre_matrix, average_ratings, movie_title_dict)


    genres_train, training_genre_preferences = process_data(train_users, training_data, movie_genre_matrix)
    test_matrix, test_genre_preferences = process_data(test_users, test_data, movie_genre_matrix)

    ratings_train = get_ratings_matrix(training_data)
    ratings_test = get_ratings_matrix(test_data)


    # epsilon list for DBSCAN
    eps_list = [0.2, 0.25, 0.3]
    min_points = [50, 100, 200]



    for eps, min_pts in zip(eps_list, min_points):
        db = DBSCAN(eps=eps, min_samples=min_pts, metric="euclidean", n_jobs=-1).fit(cluster_matrix)
        num_clusters = len(np.unique(db.labels_))
        print(f"Number of Clusters: {num_clusters}")
        cluster_dict = {}
        for i in range(num_clusters):
            cluster_dict[i] = cluster_matrix[db.labels_ == i]

        cluster_votes = Voting(num_clusters, cluster_dict)


        user_recs = make_recommendations(test_matrix, cluster_votes, genre_dict, top_10_movies_per_genre)
        predictions = predict_ratings(ratings_train, average_ratings, training_genre_preferences, movie_genre_matrix)

        MAE = mean_absolute_error(ratings_train[np.where(ratings_test > 0)], predictions[np.where(ratings_test > 0)])

        print(f"MAE: {MAE}")

        write_dict_to_file(f"top_10_movies_per_user_epsilon_{eps}_min_points_{min_pts}", user_recs)






if __name__ == "__main__":
    main()

