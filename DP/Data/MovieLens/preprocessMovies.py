import pandas as pd
import numpy as np

MOVIES_FN = "./ml-latest/movies.csv"
LINKS_FN = "./ml-latest/links.csv"
REDUCED_DATA = "MovieLens-Ratings.csv"
OUTPUT_FN = "processedMovids.csv"
COLUMNS = ["MOVIE_ID", "TITLE", "IMDB_ID", "YEAR"]

reduced = pd.read_csv(REDUCED_DATA)
movies = pd.read_csv(MOVIES_FN)
links = pd.read_csv(LINKS_FN)

movids = reduced["MovieID"].unique()


df = pd.DataFrame(columns=COLUMNS)

for _id in movids:
    # print(_id)
    # print(movies[movies["movieId"] == _id].title.to_numpy())
    title_numpy = movies[movies["movieId"] == _id].title.to_numpy()

    if np.size(title_numpy, 0) == 0:
        print("PASSING:", _id)
        print("NO TITLE IN:", title_numpy)
        continue

    title = title_numpy[0]
    index = title.rfind("(")
    name = title[: index]
    year = title[index+1: len(title)-1]
    # print(name)
    # print(year)
    imdb = links[links["movieId"] == _id].imdbId.to_numpy()[0]
    # print(imdb)
    n = len(df)
    row = [_id, name, imdb, year]
    df.loc[n] = row


# print(df)
df.to_csv(OUTPUT_FN, index=False)
