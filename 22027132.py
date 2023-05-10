# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:21:10 2023

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import scale
import sklearn.metrics as skmet

from scipy.optimize import curve_fit
import cluster_tools as ct
import scipy.optimize as opt
import errors as err

# Function to read the dataset
def read_worldbank_dataset():
    """
        This function reads in a climate change dataset from worldbank,
        and remove all unnecessary columns from the dataframe.
    """
    # Reads the csv dataset, passed as an argument into the function
    # Also skip the first four rows of the dataset
    worldbank_dataframe = pd.read_csv("worldbank_data.csv", skiprows=4)
        
    # Retrieve relevant indicators
    indicators = ['Urban population growth (annual %)',]
    dataframe = worldbank_dataframe[worldbank_dataframe['Indicator Name']
                                    .isin(indicators)]
    
    # A list of the unnecessary columns to remove from the dataframe
    columns_to_drop = ["Country Code", "Indicator Code", "Unnamed: 66"]
    
    # Remove unnecessary columns from the dataframe 
    dataframe = dataframe.drop(labels=columns_to_drop, axis=1)
        
    # Reset the index and drop the index column
    dataframe = dataframe.reset_index()
    dataframe = dataframe.drop("index", axis=1)
     
    return dataframe


worldbank_dataframe = read_worldbank_dataset()
print(worldbank_dataframe.describe())


dataframe = worldbank_dataframe[["1965", "1970", "1975", "1980", "1985", 
                                 "1990", "1995", "2010", "2020"]].copy()
print(dataframe)


# Drop rows with NaN in 2020
cleaned_dataframe = dataframe.dropna()
print(cleaned_dataframe.describe())

cleaned_dataframe = cleaned_dataframe.reset_index()
cleaned_dataframe = cleaned_dataframe.drop("index", axis=1)

print(cleaned_dataframe.corr())
ct.map_corr(cleaned_dataframe, 9)
pd.plotting.scatter_matrix(cleaned_dataframe, figsize=(9.0, 9.0))
plt.tight_layout() # helps to avoid overlap of labels
plt.show()

fitted_dataframe = cleaned_dataframe[["1970", "2010"]].copy()

corr = fitted_dataframe.corr()
print(corr)

# normalise, store minimum and maximum
normalized_dataframe, dataframe_min, dataframe_max = ct.scaler(
    fitted_dataframe)

print()
print("n score")
# loop over number of clusters
for ncluster in range(2, 7):
    # set up the clusterer with the number of expected clusters
    kmeans = KMeans(n_clusters=ncluster)
    # Fit the data, results are stored in the kmeans object
    kmeans.fit(normalized_dataframe) # fit done on x,y pairs
    labels = kmeans.labels_
    # extract the estimated cluster centres
    cen = kmeans.cluster_centers_
    # calculate the silhoutte score
    print(ncluster, skmet.silhouette_score(normalized_dataframe, labels))

ncluster = 7

# set up the clusterer with the number of expected clusters
kmeans = KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(normalized_dataframe) # fit done on x,y pairs
labels = kmeans.labels_

# cluster by cluster
plt.figure(figsize=(7.0, 7.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(normalized_dataframe["1970"], normalized_dataframe["2010"], 
            10, labels, marker="o", cmap=cm)

# extract the estimated cluster centres
cen = kmeans.cluster_centers_

xcen = cen[:, 0]
ycen = cen[:, 1]
plt.scatter(xcen, ycen, c="k", marker="d", s=45)

plt.title("Annual Urban population growth(1970 & 2010)")
plt.xlabel("1970")
plt.ylabel("2010")

plt.show()

################### UNNORMALIZED DATA ###########################
ncluster = 7

# set up the clusterer with the number of expected clusters
kmeans = KMeans(n_clusters=ncluster)

# Fit the data, results are stored in the kmeans object
kmeans.fit(fitted_dataframe) # fit done on x,y pairs
labels = kmeans.labels_

# cluster by clusterplt.figure(figsize=(7.0, 7.0))
cm = plt.cm.get_cmap('tab10')
plt.scatter(fitted_dataframe["1970"], fitted_dataframe["2010"], 
            10, labels, marker="o", cmap=cm)

# extract the estimated cluster centres
cen = kmeans.cluster_centers_

xcen = cen[:, 0]
ycen = cen[:, 1]
plt.scatter(xcen, ycen, c="k", marker="d", s=45)

plt.title("Annual Urban population growth(1970 & 2010)")
plt.xlabel("1970")
plt.ylabel("2010")

plt.show()

############################################################################

dataframe2 = worldbank_dataframe.T.reset_index()
print(dataframe2)

dataframe2.columns = dataframe2.iloc[0]
dataframe2 = dataframe2.drop(0)

dataframe2 = dataframe2[["Country Name", "United States"]]

dataframe2 = dataframe2.drop(1)

dataframe2 = dataframe2.rename(
    columns={"Country Name": "Years", 
             "United States": "Growth(%)"})

# drop rows with NaN values
dataframe2 = dataframe2.dropna().reset_index()

dataframe2 = dataframe2.drop("index", axis=1)

dataframe2.plot("Years", "Growth(%)")

print(dataframe2)
