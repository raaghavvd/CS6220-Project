import pandas as pd
import numpy as np
import re
from io import StringIO
import os


def create_raw_data():
    files_list = ["combined_data_1.txt", "combined_data_2.txt", "combined_data_3.txt", "combined_data_4.txt"]
    frames = []
    for file in files_list:
        # Update this path to wherever you stored the combined data files
        fd = open(f"C:\\Users\\mmeuse\\Mark Docs\\CS6220 Data Mining\\Project\\archive\\{file}")
        cur_file = fd.read()

        # pattern for splitting the document by movie example: "1:" or "22:"
        split_pattern = re.compile(r"\d+:\n")

        # This matches just the number before the colon
        movie_id_pattern = re.compile(r"\d+(?=:)")

        # split the file
        split_results = re.split(split_pattern, cur_file)

        # remove the first element of the list after split as it is an empty string.
        split_results = split_results[1:]

        # Find the first movie id in the file, then increment movie id for each set of ratings found
        movie_id = int(re.match(movie_id_pattern, cur_file).group())

        # Iterate through the splits, read them into a pandas dataframe in memory and add the movie id as a column
        for group in split_results:
            df = pd.read_csv(StringIO(group))
            df.columns = ["user_id", "rating", "time_stamp"]
            df["movie_id"] = movie_id
            frames.append(df)
            movie_id += 1

        # concatenate all the pandas dataframes
        ratings_matrix = pd.concat(frames)

        # remove the timestamp column
        ratings_matrix = ratings_matrix.drop(columns="time_stamp")

        # output the combined file as a csv
        if not os.path.isfile('netflix_condensed.csv'):
            ratings_matrix.to_csv('netflix_condensed.csv', header='column_names', index=False)
        else:
            ratings_matrix.to_csv('netflix_condensed.csv', mode='a', header=False, index=False)



def update_user_rating_matrix(raw_data_matrix, user_rating_matrix, index_map):
    for row in raw_data_matrix:
        row_index = index_map[row[0]]
        col_index = row[2] - 1
        user_rating_matrix[row_index][col_index] = row[1]


def main():

    # Parses the combined data files and generates a single input file. Only needs to be executed once
    create_raw_data()

    unique_movies_count = 17770
    unique_users_count = 480189
    user_movie_matrix = np.zeros((unique_users_count, unique_movies_count))
    raw_data_chunks = pd.read_csv("netflix_condensed.csv", chunksize=100000)
    id_dict = {}
    index = 0
    for chunk in raw_data_chunks:
        user_ids = chunk["user id"].unique()
        for id in user_ids:
            if id not in id_dict:
                id_dict[id] = index
                index += 1
        chunk_numpy = chunk.to_numpy()
        update_user_rating_matrix(chunk_numpy, user_movie_matrix, id_dict)

    df = pd.DataFrame(user_movie_matrix)
    df.to_csv("netflix_user_ratings.csv", index=False)




if __name__ == "__main__":
    main()

# for chunk in raw_data_chunks:
#     df = chunk.drop(columns="time stamp")
#     if not os.path.isfile('netflix_condensed.csv'):
#         df.to_csv('netflix_condensed.csv', header='column_names', index=False)
#     else:
#         df.to_csv('netflix_condensed.csv', mode='a', header=False, index=False)







