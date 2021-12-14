import pandas as pd
import math

movieFN = "./ml-10M100k/ratings.dat"


df = pd.read_csv(movieFN, sep="::")
timestamps = df["Timestamp"].unique().tolist()
timestamps.sort()
cutoff = timestamps[math.floor(len(timestamps) * 0.8)]
print(cutoff)

train = df[df["Timestamp"] < cutoff]
test = df[df["Timestamp"] >= cutoff]

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

train.to_csv("ml-10m.train.csv", index=False)
test.to_csv("ml-10m.test.csv", index=False)
