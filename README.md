# CS6220-Project


## Authors: Chris Curtis, Raaghavv Devgon, Mark Meuse



File Descriptions.


MovieStats.ipynb- Contains the code for data visualization on MovieLens 1M
NN_Movie.ipynb- Contains the code for the Autoencoders
Imdbinfo.ipynb- Contains the code for the imdb metadata dumper.


DP/Data

HandlePredictions.ipynb - used to collect and process the partial results of the survey paper 
PREDICTIONS.py - contains the predictions made by the survey paper
getNaivePrediction.py - contains naive implementation
makeSamples - creates the test and train split
reviewConsolidation - database reducing script

Ml-1m2 - the train and test files

DP/Data/MovieLens

preprocessMovies.py - creates a csv to aid in the expanded database processedMovides.csv - the csv used for processing in the expanded database

DP/JobArrayTesting

Clear.sh - utility script for cleaning results directory runinference.py - the implementation for the survey paper slurm-test.sh - the slurm script used for collecting results on the test set using batches



find_optimal_clusters.py produces the SSE vs Cluster count image used in the report 

hopkins_statistic.py contains the function that was used to calculate the hopkins statistic used in the report

voting_theory.py is the original implementation of the voting theory paper

improved_voting_theory.py is the improved implementation of the voting theory paper.
