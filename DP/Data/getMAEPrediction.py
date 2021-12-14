import pandas as pd

train = pd.read_csv("ml-1m2.train.csv")
test = pd.read_csv("ml-1m2.test.csv")

MEANS = {}

movids = test["MovieID"].unique().tolist()
for mid in movids:
    MEANS[mid] = train[train["MovieID"] == mid]["Rating"].mean()


def Classify(n):
    return 1 if n >= 4 else 0


MAE = 0
cntr = 0
tp = 0
fp = 0
fn = 0

for row in test.iterrows():
    row = row[1]
    mid = row.MovieID
    rating = row.Rating

    val = abs(rating - MEANS[mid])
    if pd.isna(val):  # When there is no history do not offer a prediction
        continue
    cntr += 1
    MAE += val

    true = Classify(rating)
    pred = Classify(MEANS[mid])

    #print(true, pred)

    if true == 1 and pred == 1:
        tp += 1

    if true == 1 and pred == 0:
        fn += 1

    if true == 0 and pred == 1:
        fp += 1

    # break


MAE /= cntr
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print(MAE, precision, recall)
