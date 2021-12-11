import pandas as pd
import math

from pandas.core.algorithms import unique

DIR = "./samples"

for i in range(3):
    fn = DIR + "/sample-" + str(i) + ".csv"
    df = pd.read_csv(fn)
    timestamps = df["timestamp"].unique().tolist()
    timestamps.sort()
    cutoff = timestamps[math.floor(len(timestamps) * 0.8)]
    print(cutoff)

    train = df[df["timestamp"] < cutoff]

    test = df[df["timestamp"] >= cutoff]

    MAE = {}
    movids = test["MovieID"]

    uniqueMovies = movids.unique().tolist()
    for item in uniqueMovies:
        ratings = train[train["MovieID"] == item]["Rating"]
        mean = ratings.mean()
        MAE[item] = mean

    means = []
    for id in movids:
        means.append(MAE[id])
    test["Mean-Prediction"] = means
    train.to_csv(DIR + "/sample-" + str(i) + ".train.csv")
    test.to_csv(DIR + "/sample-" + str(i) + ".test.csv")
