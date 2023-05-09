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
from scipy.optimize import curve_fit

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
    indicators = ['Urban population growth (annual %)', 
                  'Arable land (% of land area)']
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


dataframe = read_worldbank_dataset()
print(dataframe.describe())

# Drop rows with NaN in 2020
cleaned_2020_df = dataframe[dataframe["2020"].notna()]
print(cleaned_2020_df.describe())

cleaned_2020_df_ = cleaned_2020_df[["Indicator Name", "1998", "2000", 
                                    "2018", "2020"]].copy()
print(cleaned_2020_df_.describe())

#grouped = cleaned_2020_df_.groupby('Indicator Name')

#print(grouped.describe().T)


