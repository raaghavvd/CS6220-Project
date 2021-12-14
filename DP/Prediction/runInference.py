import math
import numpy as np
import pandas as pd
from datetime import datetime
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--start_num', help='start index')
parser.add_argument('--end_num', help='end index')
args = parser.parse_args()

TRAIN_DF = pd.read_csv("/scratch/curtis.ch/reducedRatings-10MIll-S1-Train.csv")
TEST_DF = pd.read_csv("/scratch/curtis.ch/reducedRatings-10MIll-S1-Test.csv")
UIDS = TRAIN_DF["UserID"].unique().tolist()

RATINGS = {}
MOVIES = {}
PEARSON_COEFFICENT_PAIRS = {}
AVERAGES = {}

def GetUserReviews(uid):
    return TRAIN_DF[TRAIN_DF["UserID"] == uid]


def PCHelper(movidA, movidB):
    reviewsA = GetUserReviews(movidA)
    reviewsB = GetUserReviews(movidB)

    usersA = reviewsA["MovieID"].tolist()
    usersB = reviewsB["MovieID"].tolist()

    usersInCommon = {}

    cntr = 0
    sumA = 0
    sumB = 0
    for uid in usersA:
        if uid in usersB:
            RatingA = reviewsA[reviewsA["MovieID"] == uid]["Rating"].iloc[0]
            RatingB = reviewsB[reviewsB["MovieID"] == uid]["Rating"].iloc[0]

            sumA += RatingA
            sumB += RatingB
            cntr += 1

            usersInCommon[uid] = (RatingA, RatingB)

    if cntr == 0:
        return {}, 0, 0

    return usersInCommon, sumA/cntr, sumB/cntr


def GetAverageRating(uid):
    if not uid in AVERAGES:
        avg = TRAIN_DF[TRAIN_DF["UserID"] == uid]["Rating"].mean()
        AVERAGES[uid] = avg
        return avg
    return AVERAGES[uid]


def GetMovies(uid):
    if not uid in MOVIES:
        movies = TRAIN_DF[TRAIN_DF["UserID"] == uid]["MovieID"].tolist()
        MOVIES[uid] = movies
        return movies
    return MOVIES[uid]

def GetRating(uid, movid):
    if not (uid, movid) in RATINGS:
        Rating = TRAIN_DF[TRAIN_DF["UserID"] == uid][TRAIN_DF["MovieID"] == movid]["Rating"].iloc[0]
        RATINGS[(uid, movid)] = Rating
        return Rating
    return RATINGS[(uid, movid)]

def GetPearsonCorrelation(uidA, uidB):
    _min = uidA if uidA < uidB else uidB
    _max = uidB if _min == uidA else uidA
    
    if (_min, _max) in PEARSON_COEFFICENT_PAIRS:
        return PEARSON_COEFFICENT_PAIRS[ (_min, _max) ] 
    
    usersInCommon, avgA, avgB = PCHelper(uidA, uidB)

    if len(usersInCommon) == 0:
        return None

    numerator = 0
    denomA = 0
    denomB = 0

    for item in usersInCommon:
        Ratings = usersInCommon[item]
        numerator += ( (Ratings[0] - avgA) * (Ratings[1] - avgB) )
        denomA += (Ratings[0] - avgA) ** 2
        denomB += (Ratings[1] - avgB) ** 2

    if numerator == 0:
        return 0

    denomA = math.sqrt(denomA)
    denomB = math.sqrt(denomB)
    pearsonCoeff = numerator / (denomA * denomB)
    return pearsonCoeff

def Predict(uid, movid, debug=False):
    first_time = datetime.now()
    #if debug:
    #    print("Collecting average Rating for", uid)
    avg = GetAverageRating(uid)

    num = 0
    denom = 0
    #print("Making weighted sum...")
    for id in UIDS:
        if not movid in GetMovies(id):
            continue

        weight = GetPearsonCorrelation(uid, id)
        if weight is None or weight == 0:
            continue
        
        Rating = GetRating(id, movid)
        otherAvg = GetAverageRating(id)
        
        num += (Rating - otherAvg) * weight
        denom += abs(weight)
        
    later_time = datetime.now()
    duration = later_time - first_time
    if debug:
        print("Total time in seconds", duration.total_seconds())
    
    if num == 0:
        return avg, duration.total_seconds()
    return avg + (num / denom), duration.total_seconds()

def GetTestTuples():
    start = int(args.start_num) - 1
    end = int(args.end_num) - 1
    print("Running", start, "through", end)
    uids = TEST_DF["UserID"].tolist()[start: end]
    movids = TEST_DF["MovieID"].tolist()[start: end]
    tuples = []
    for i in range(len(uids)):
        tuples.append( (uids[i], movids[i]) )
    return tuples

print("PROGRAM START")
testTuples = GetTestTuples()
predictions = []
times = []
for item in testTuples:
    val, time = Predict(item[0], item[1], True)
    predictions.append(val)
    times.append(time)

totalTime = 0
for time in times:
    totalTime += time
print("Total:", totalTime)
print("Avg:", totalTime / totalTime)
print("Predictions:", predictions)
