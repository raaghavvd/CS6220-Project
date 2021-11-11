import pandas as pd
import numpy as np



def update_user_rating_matrix(raw_data_matrix, user_rating_matrix, index_map):
    for row in raw_data_matrix:
        row_index = index_map[row[1]]
        col_index = row[0] - 1
        user_rating_matrix[row_index][col_index] = row[2]


def main():

    unique_movies_count = 17770
    unique_users_count = 480189
    user_movie_matrix = np.zeros((unique_users_count, unique_movies_count))
    raw_data = pd.read_csv("consolidatedReviews.csv")

    id_dict = {}
    index = 0
    user_ids = raw_data["UserID"].unique()
    for id in user_ids:
        if id not in id_dict:
            id_dict[id] = index
            index += 1
    update_user_rating_matrix(raw_data.to_numpy(), user_movie_matrix, id_dict)

    df = pd.DataFrame(user_movie_matrix)
    df.to_pickle("netflix_user_ratings.pkl.compress", compression="gzip")







if __name__ == "__main__":
    main()
