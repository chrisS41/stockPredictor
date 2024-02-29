# stockPredictor
stock prediction model with CNN(Convolutional Neural Network)
<br><br>

## About
College project that using four models (ARIMA, Linear Regression, LSTM and CNN) to predict S&P 500 price movement for the next day.

Compare the models and decide which is the best for time-series prediction.

This repository starts with CNN model only.
<br><br>

## Requirements
### [Python v3.8](https://www.python.org/downloads/release/python-3813/)<br>
### [anaconda3](https://www.anaconda.com/download)<br>
packages have to be installed in anaconda environment.<br>
<ul>
  <li>numpy</li>
  <li>pandas</li>
  <li>matplotlib</li>
  <li>sklearn</li>
  <li>keras</li>
</ul>
<br>

## Dataset
The dataset in this repository is a historical data of S&P 500 ETF from August 2016 to April 2021.
<br><br>

## Result
![bestModel](https://github.com/chrisS41/stockPredictor/assets/87973935/978a69da-1596-4ebf-a215-be2ce15c880e)
![CNNresult](https://github.com/chrisS41/stockPredictor/assets/87973935/15064ba5-7f78-42c0-b96b-fb3819a5d52b)

In the result, the validation loss is around 0.05% and accuracy is 1.12%.<br>
The prediction result was moved to the right, so the accuracy was evaluated lower.<br>
The expected causes are as follows.
<ul>
  <li>Overfitting</li>
  <li>Bad noise reduction</li>
</ul>
<br><br><br>
