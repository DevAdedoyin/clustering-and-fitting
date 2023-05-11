# -*- coding: utf-8 -*-
"""
Created on Tue May  9 13:21:10 2023

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans
#from sklearn.preprocessing import scale
import sklearn.metrics as skmet

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

# Reads the csv dataset, passed as an argument into the function
# Also skip the first four rows of the dataset
gdp_dataframe = pd.read_csv("gdp_data.csv", skiprows=4)
    
# A list of the unnecessary columns to remove from the dataframe
columns_to_drop = ["Country Code", "Indicator Code",]

# Remove unnecessary columns from the dataframe 
gdp_dataframe = gdp_dataframe.drop(labels=columns_to_drop, axis=1)
    
# Reset the index and drop the index column
gdp_dataframe = gdp_dataframe.reset_index()
gdp_dataframe = gdp_dataframe.drop("index", axis=1)

gdp_data = gdp_dataframe.T.reset_index()
print(gdp_data)

gdp_data.columns = gdp_data.iloc[0]
gdp_data = gdp_data.drop(0)

gdp_data = gdp_data[["Country Name", "United States"]]

gdp_data = gdp_data.drop(1)

gdp_data = gdp_data.rename(
    columns={"Country Name": "Years", 
             "United States": "GDP per capita"})

# drop rows with NaN values
gdp_data = gdp_data.dropna().reset_index()

gdp_data = gdp_data.drop("index", axis=1)

def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f

param, covar = opt.curve_fit(logistics, gdp_data["Years"].astype(int), 
                            gdp_data["GDP per capita"], p0=(15000, 0.04, 1980))
print("Fit parameter", param)

sigmas = np.sqrt(np.diag(covar))

gdp_data["fit"] = logistics(gdp_data["Years"].astype(int), *param)
plt.figure()
plt.title("logistics function")
plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], label="data")
plt.plot(gdp_data["Years"], gdp_data["fit"], label="fit")

years_list = ["1960", "1970", "1980", "1990", "2000", "2010", "2020"]

plt.xticks(years_list)
plt.legend()
plt.show()

# extract variances and calculate sigmas
#gdp_data["trial"] = logistics(gdp_data["Years"].astype(int), 15000, 0.02, 1990)

# call function to calculate upper and lower limits with extrapolation
# create extended year range
years_ = np.arange(1960, 2031)
forecast = logistics(years_, *param)
plt.figure()
plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], label="GDP per capita")
plt.plot(years_, forecast, label="forecast")

plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()

lower, upper = err.err_ranges(years_, logistics, param, sigmas)

plt.figure()
plt.title("logistics function")
plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], label="GDP per capita")
plt.plot(years_, forecast, label="fit")

plt.xticks(years_list)

# plot error ranges with transparency
plt.fill_between(years_, lower, upper, alpha=0.7, color='green')

plt.legend(loc="upper left")
plt.show()

print("Population in")
print("2030:", logistics(2030, *param) / 1000, "Mill.")
print("2040:", logistics(2040, *param) / 1000, "Mill.")
print("2050:", logistics(2050, *param) / 1000, "Mill.")
print(gdp_data)

print("Fit parameter", param)
