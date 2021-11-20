from sklearn.cluster import DBSCAN
import pandas as pd
import numpy as np
import random
import json
from sklearn.neighbors import NearestNeighbors


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


def create_user_genre_vector(d, num_genres):
    v = np.zeros((d,), dtype="float16")
    genre_index_list = select_genres(range(d), num_genres)
    for index in genre_index_list:
        val = np.float16(random.random())
        v[index] = val
    v = v / np.sum(v)
    return v


def create_training_matrix(n, d=19, num_genres=4):
    m = np.empty((n, d))
    for i in range(n):
        m[i] = create_user_genre_vector(d, num_genres)
    return m


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
    return np.apply_along_axis(np.sum, 0, cluster)



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
    genre_map = db.iloc[:, -19:]
    return genre_map


def get_movie_recommendations(user, top_movies_per_genre):
    user_genre = user.argmax()
    return top_movies_per_genre[user_genre]


def get_top_10_movies_per_genre(genre_dict, movie_genres, average_ratings, movie_titles):
    top10_dict = {}
    assert len(movie_genres.iloc[:, 0]) == len(average_ratings)
    for i in range(len(movie_genres.iloc[:, 0])):
        all_genres_for_movie = movie_genres.iloc[i, :].values
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
    return db.to_dict()



def main():
    movie_title_dict = get_movie_title_dict()
    genre_dict = get_genre_dict()
    movie_genres = get_movie_genres()
    average_ratings = calculate_average_movie_ratings()
    top_10_movies_per_genre = get_top_10_movies_per_genre(genre_dict, movie_genres, average_ratings, movie_title_dict)
    # number of users
    n = 60000
    # number of genres
    d = 19
    # epsilon list for DBSCAN
    eps_list = [0.5, 0.75, 1]
    min_points = [100, 150, 200]
    train_matrix = create_training_matrix(n, d)
    test_matrix = create_training_matrix(int(n/4), d)

    for eps, min_pts in zip(eps_list, min_points):
        # neighbor_obj = NearestNeighbors(radius=eps)
        # neighbor_obj.fit(train_matrix)
        # nieghbors_graph = neighbor_obj.radius_neighbors_graph(train_matrix, mode="distance")
        # db = DBSCAN(eps=eps, min_samples=min_pts, metric="precomputed", n_jobs=-1).fit(nieghbors_graph)
        db = DBSCAN(eps=eps, min_samples=min_pts, metric="euclidean", n_jobs=-1).fit(train_matrix)
        num_clusters = len(np.unique(db.labels_))
        cluster_dict = {i: train_matrix[db.labels_ == i] for i in range(num_clusters)}
        # core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
        # core_samples_mask[db.core_sample_indices_] = True
        # labels = db.labels_

        cluster_votes = Voting(num_clusters, cluster_dict)
        user_recs = make_recommendations(test_matrix, cluster_votes, genre_dict, top_10_movies_per_genre)
        write_dict_to_file(f"top_10_movies_per_user_{eps}", user_recs)






if __name__ == "__main__":
    main()

