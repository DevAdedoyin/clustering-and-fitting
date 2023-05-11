# -*- coding: utf-8 -*-
"""
Created on Thu May 11 13:13:39 2023

@author: DELL
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import scipy.optimize as opt
import errors as err


def read_gdp_dataset():
    """
        This function reads in the gdp dataset from the worldbank data.
        It selects the necessary columns needed and removes unnecesary columns.
        The function returns a dataframe containing gdp per capita of the 
        United States from the year 1960 to 2021.
    """
    
    # Reads the csv dataset, passed as an argument and skip the first four 
    # rows of the dataset
    gdp_dataframe = pd.read_csv("gdp_data.csv", skiprows=4)
        
    # A list of the unnecessary columns to remove from the dataframe
    columns_to_drop = ["Country Code", "Indicator Code",]
    
    # Remove unnecessary columns from the dataframe 
    gdp_dataframe = gdp_dataframe.drop(labels=columns_to_drop, axis=1)
        
    # Reset the index and drop the index column
    gdp_dataframe = gdp_dataframe.reset_index()
    gdp_dataframe = gdp_dataframe.drop("index", axis=1)
    
    # Transpose and reset index of the dataframe
    gdp_data = gdp_dataframe.T.reset_index()
    
    # Drop the row not needed
    gdp_data.columns = gdp_data.iloc[0]
    gdp_data = gdp_data.drop(0)
    
    # Select the Country name and United States column from the transposed 
    # dataframe and drop row one
    gdp_data = gdp_data[["Country Name", "United States"]]
    gdp_data = gdp_data.drop(1)
    
    # Rename column names
    gdp_data = gdp_data.rename(columns={"Country Name": "Years", 
                 "United States": "GDP per capita"})
    
    # Drop rows with NaN values and reset index and drop the index column
    gdp_data = gdp_data.dropna().reset_index()
    gdp_data = gdp_data.drop("index", axis=1)

    return gdp_data

# Call the read_gdp_dataset function
gdp_data = read_gdp_dataset()

# Compute the logistic function
def logistics(t, a, k, t0):
    """ Computes logistics function with scale and incr as free parameters
    """
    f = a / (1.0 + np.exp(-k * (t - t0)))
    return f

# Create a curve fit from the dataframe using the curve_fit function
param, covar = opt.curve_fit(logistics, gdp_data["Years"].astype(int), 
                            gdp_data["GDP per capita"], p0=(50000, 0.01, 1970))
print("Fit parameter", param)

sigmas = np.sqrt(np.diag(covar))

gdp_data["fit"] = logistics(gdp_data["Years"].astype(int), *param)
plt.figure()

# Title of the plot and plot of the fitted data
plt.title("Logistics function for GDP per capita")
plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], label="data")
plt.plot(gdp_data["Years"], gdp_data["fit"], label="fit")

years_list = ["1960", "1970", "1980", "1990", "2000", "2010", "2020"]

plt.xticks(years_list)
plt.legend()
plt.show()

# call function to calculate upper and lower limits with extrapolation
# create extended year range
year = np.arange(1960, 2041)
forecast = logistics(year, *param)

year_list_forecast = years_list + ["2030", "2040"]
print(year_list_forecast)

forecast_df = pd.DataFrame({"forecast": forecast, "Years": year.astype(str)})
gdp_data = pd.merge(gdp_data, forecast_df, on="Years", how="outer")



print(gdp_data)

#gdp_data = gdp_data.fillna(0, inplace=True)

plt.figure()
plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], 
         label="GDP")
plt.plot(gdp_data["Years"], gdp_data["forecast"], 
         label="forecast")

plt.xticks(year_list_forecast)

plt.xlabel("year")
plt.ylabel("GDP")
plt.legend()
plt.show()

lower, upper = err.err_ranges(gdp_data["Years"].astype(int), 
                              logistics, param, sigmas)

plt.figure()
plt.title("logistics function")

plt.plot(gdp_data["Years"], gdp_data["GDP per capita"], label="GDP per capita")
plt.plot(forecast, label="fit")

# plot error ranges with transparency
plt.fill_between(gdp_data["Years"], lower, upper, alpha=0.7, color='green')

plt.xticks(year_list_forecast)

plt.legend(loc="upper left")
plt.show()

print("Population in")
print("2030:", logistics(2030, *param) / 1000, "Mill.")
print("2040:", logistics(2040, *param) / 1000, "Mill.")
print("2050:", logistics(2050, *param) / 1000, "Mill.")
print(gdp_data)

print("Fit parameter", param)