import pandas as pd
import math

from sklearn.model_selection import train_test_split

movieFN = "ratings.csv"


df = pd.read_csv(movieFN)
df.sort_values(by='Timestamp', ascending=True, inplace=True)
train, test = train_test_split(df, test_size=0.2, shuffle=False)
train.to_csv("ml-1m2.train.csv", index=False)
test.to_csv("ml-1m2.test.csv", index=False)
