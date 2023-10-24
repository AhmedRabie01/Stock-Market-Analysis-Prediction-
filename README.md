# Stock Market Predictions

 This a End To End project, we aim to build a machine learning model that predicts the stock prices of Microsoft. The data used in this project was collected from Yahoo Finance, and the analysis and predictions were made using Python and its libraries.


# project's Power BI Dashboard screen shots
![project web app]([https://github.com/AhmedRabie01/Stock-Market-Analysis-Prediction-/blob/main/photo/2023-10-17 (2).png](https://github.com/AhmedRabie01/Stock-Market-Analysis-Prediction-/blob/12ef23efef7fc403bb186f8dcfaa8db08efa95e8/photo/2023-10-17%20(2).png))

![project prediction]([https://github.com/AhmedRabie01/Stock-Market-Analysis-Prediction-/blob/main/photo/2023-10-17 (9).png](https://github.com/AhmedRabie01/Stock-Market-Analysis-Prediction-/blob/12ef23efef7fc403bb186f8dcfaa8db08efa95e8/photo/2023-10-17%20(9).png))


# Data Collection:

The data used in this project consists of historical stock prices for the four companies from January 1st, 2010 to December 31st, 2023. The data was collected from Yahoo Finance using the yfinance library in Python.

# Data Preprocessing:

Before we could use the data, we had to preprocess it. This involved converting the date column to a datetime format, removing any missing values, and creating new features such as moving averages, relative strength index, and momentum indicators.

# Exploratory Data Analysis:

Next, we conducted exploratory data analysis to gain insights into the data. We plotted the stock prices of the four companies over time, and we also looked at their daily returns, volatility, and correlations. We found that the four companies have positive correlations, meaning that they tend to move in the same direction.


# Model Building and Prediction:

To build our prediction model, we used a machine learning algorithm called the Long Short-Term Memory (LSTM) neural network. We trained the model on the historical stock prices for the four companies and made predictions for the future stock prices of Microsoft.

# Evaluation:

To evaluate the performance of our model, we used the mean squared error (MSE) as our metric. We compared the predicted stock prices to the actual stock prices for Microsoft, and we found that our model had a low MSE, indicating that it performed well in making predictions.

# Conclusion:

In conclusion, we successfully built a machine learning model that predicts the stock prices of Google, Microsoft, Apple, and Amazon. Our model had a low MSE when predicting the stock prices for Microsoft, indicating that it can be used to make accurate predictions for the other companies as well.
