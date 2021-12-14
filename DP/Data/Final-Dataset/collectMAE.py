import pandas as pd

fn = "ml-10m.test.csv"

df = pd.read_csv(fn)
print(df)

actuals = df["Rating"].tolist()
predicited = df["Mean-Prediction"].tolist()
n = len(actuals)
MAE = 0

for i in range(len(actuals)):
    if pd.isna(predicited[i]):
        continue

    val = abs(actuals[i] - predicited[i])
    MAE += val

MAE /= n
print(MAE)
