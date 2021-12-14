import pandas as pd
import json

COLUMNS = ["Uid", "Training-Mean", "Runtime",
           "IMDB-Rating", "Votes", "Awards", "Timestamp"]

train = pd.read_csv("ml-1m.train.csv")
test = pd.read_csv("ml-1m.test.csv")
_dict = None

with open("movieLens_1M_dump.txt") as json_file:
    _dict = json.load(json_file)

train_movids = train["MovieID"].unique().tolist()
test_movids = test["MovieID"].unique().tolist()

for _id in train_movids:
    if not str(_id) in _dict:
        train = train[train.MovieID != _id]

for _id in test_movids:
    if not _id in _dict:
        test = test[test.MovieID == _id]


newTrainData = []
for index, row in train.iterrows():
    movid = row.MovieID
    uid = row.UserID
    mean = train[train["MovieID"] == movid]["Rating"].mean()
    data = _dict[str(movid)]
    runtime = data["Runtime"]
    imbd = data["IMDB Ratings"]
    votes = data["IMDB Votes"]
    awards = data["Awards"]
    timestamp = row.Timestamp
    newRow = [uid, mean, runtime, imbd, votes, awards, timestamp]
    newTrainData.append(newRow)
    #i = len(newTrain)
    #newTrain[i] = newRow
newTrain = pd.DataFrame(newTrainData, columns=COLUMNS)
#newTest = pd.DataFrame(columns=COLUMNS)
# print(newTrain)
newTrain.to_csv("Expanded-Train.csv", index=False)
