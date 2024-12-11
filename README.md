# A Comparison of Machine Learning Methods to Forecast Tropospheric Ozone Levels in Delhi

In order to run this code you need to have python installed with the libraries numpy, matplotlib, pandas, and sklearn. Then run the annual analysis with:

    cd annual_model
    jupyter notebook annual_model.ipynb

Seasonal models are similar.

## Abstract

Ground-level ozone is a pollutant that is harmful to urban populations, particularly in developing countries where it is present in significant quantities. It greatly increases the risk of heart and lung disease and harms agricultural crops. This study hypothesized that, as a secondary pollutant, ground-level ozone is amenable to 24-hour forecasting based on measurements of weather conditions and primary pollutants such as nitrogen oxides and volatile organic compounds.
We developed software to analyze hourly records of 12 air pollutants and 5 weather variables over the course of one year in Delhi, India. To determine the best predictive model, eight machine learning algorithms were tuned, trained,  tested, and compared using cross-validation with hourly data for a full year. The algorithms, ranked by R2 values, were XGBoost (0.61), Random Forest (0.61), K-Nearest Neighbors Regression (0.55), Support Vector Regression (0.48), Decision Trees (0.43), AdaBoost (0.39), linear regression (0.39).
When trained by separate seasons across five years, predictive capabilities of all models increased, with a maximum R2 of 0.75 during winter. Bidirectional-Long Short-Term Memory was the least accurate model for annual training, but had some of the best predictions for seasonal training.  Out of five air quality index categories, the XGBoost model was able to predict the correct category 24 hours in advance 90% of the time when trained with full-year data. Separated by season, winter is considerably more predictable (97.3%), followed by post-monsoon (92.8%), monsoon (90.3%), and summer (88.9%). These results show the importance of training machine learning methods with season-specific data sets and comparing a large number of methods for specific applications.
