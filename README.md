# Cluster Analysis of Urban Population Growth: A Comparative Study of Economic Trends in Different Countries and Fitting of a Logistic Model for GDP per Capita in the USA

# By Adedoyin Idris
# Module Team Include: Ralf Napiwotzki (module leader), Kasia Nowak, Matt Doherty

# Abstract
The aim of this project topic was to analyze the relationship between urban population growth and GDP per capita for various countries using clustering and curve fitting methods. The clustering analysis identified distinct clusters based on urban population growth, while the curve fitting analysis showed a positive correlation between GDP per capita and time. These findings can provide insights into the economic and environmental trends of different countries.

# Introduction
* The impact of climate change on our planet has become a major concern in recent years, and its effects are increasingly felt in urban areas where the majority of the world's population lives.
* Urbanization has been identified as a key factor contributing to climate change, with the rapid growth of urban populations leading to increased greenhouse gas emissions and greater vulnerability to climate-related risks.
* At the same time, economic growth is seen as an important driver of sustainable development, providing resources to mitigate and adapt to climate change.
* In this context, it is important to explore the relationship between urbanization and economic growth, and to identify patterns and trends that can inform policy and planning decisions.
* This poster presents an analysis of the relationship between urbanization and economic growth based on clustering and fitting techniques applied to data from the World Bank.

# The Dataset
* The dataset being used is the World Bank Climate Change dataset, which contains various indicators related to climate change and its impacts on different countries. 
* Some of the indicators include greenhouse gas emissions, temperature changes, forest area, and more. 
* The data spans multiple decades and covers a large number of countries, making it a valuable resource for analyzing global trends and identifying patterns in climate change. 
* In this particular analysis, the focus is on two indicators: Urban population growth (annual %) for clustering and GDP per capita for fitting.

# Clustering Methodology and Result
* Cluster analysis was performed on the dataset using the K-means algorithm to divide countries into four distinct clusters based on their Urban population growth (annual %) values. 
* The countries were clustered based on the similarity of their urban population growth rate over time. 
* The clustering analysis revealed four distinct clusters of countries with similar growth patterns.

# Using the unnormalized clustering result from the plot below, it can be seen that
* The Cluster 0 includes countries with consistently high urban population growth rates, such as Afghanistan, Angola, and Congo. 
* The Cluster 1 contains countries with a consistently low urban population growth rate, such as Greece, Italy, and Japan. 
* The Cluster 2 includes countries with a fluctuating urban population growth rate, such as Brazil, China, and India. 
* The Cluster 3 cluster consists of countries with an initially high urban population growth rate that has gradually decreased over time, such as Egypt, Kenya, and Uganda.

# Unnormalized
![unnormalized](https://github.com/DevAdedoyin/clustering-and-fitting/assets/59482569/f6ef3266-3bda-42a8-920f-920453c5de46)

# Normalized
![normalized](https://github.com/DevAdedoyin/clustering-and-fitting/assets/59482569/49e610c3-d7c3-40d3-a542-cd370806b57e)

# Fitting
* For the fitting analysis, the GDP per capita data  for the United States was used and applied a logistic function to fit the data.
* The logistic function was able to fit the GDP per capita data for the United States reasonably well, with a good coefficient of determination. 
* Using this fitted model, predictions for the next 20 years was made, and alse, the confidence range using the err_ranges function was calculated .
* The plot of the best-fitting function and the confidence range showed that the GDP per capita in the United States is expected to continue to increase, but at a slower rate than in the past. 
* This trend is consistent with what we might expect to see in a mature economy like the United States, where the growth rate tends to stabilize after a period of rapid development.
* Overall, the fitting analysis suggests that the logistic function is a useful model for describing the growth of the United States' GDP per capita and that it can be used to make reasonable predictions for the future. 

# Fitting
![fitting1](https://github.com/DevAdedoyin/clustering-and-fitting/assets/59482569/e23b6eaa-df10-4977-a502-020dae8fad14)

# Prediction
![prediction](https://github.com/DevAdedoyin/clustering-and-fitting/assets/59482569/50506b90-e266-4af0-8bec-bbd992a67e17)

# Prediction and Error Range
![error](https://github.com/DevAdedoyin/clustering-and-fitting/assets/59482569/c35e3447-101c-4eea-8a41-832ff48d9d80)

# Conclusion
* The clustering analysis provides insights into how different countries are experiencing urbanization.
* The fitting analysis shows a promising future for the USA's economic growth.
* These results demonstrate the usefulness of clustering and fitting techniques in analyzing complex datasets.
* In conclusion, our analysis shows that clustering and fitting can provide valuable insights into global trends related to climate change and economic development.
* By clustering countries based on their urban population growth rates, we were able to identify distinct groups of countries with similar trends and patterns. 
* We then fit a simple logistic function model to the GDP per capita of the USA to make predictions about their future economic growth. 
* These results can help policymakers and stakeholders better understand the complex relationships between urbanization, economic development, and climate change, and make more informed decisions about sustainable development. 
* Overall, our study demonstrates the power of data analysis and modeling for gaining insights into global challenges and developing evidence-based solutions.

# Credits
# Please, if you're going to use any part of this write up or project ensure you give credit to Adedoyin Idris

Thanks.
Kind Regards.
