# -*- coding: utf-8 -*-
"""
Created on Fri Aug 21 16:02:52 2020

@author: Acer
"""

# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the training set
dataset_train = pd.read_csv('Google_Stock_Price_Train.csv')
training_set = dataset_train.iloc[:, 1:2].values

# Feature Scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)

# Getting the inputs and outputs
X_train = training_set_scaled[0:1257]
Y_train = training_set_scaled[1:1258]

# Reshaping
X_train = np.reshape(X_train, (1257, 1, 1))




# Part 2 - Building the RNN

# Importing the Keras libraries and packages
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM

# Initialising the RNN
regressor = Sequential()

# Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units = 4, activation = 'sigmoid', input_shape = (None, 1)))

# Adding the output layer
regressor.add(Dense(units = 1))

# Compiling the RNN
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')

regressor.summary()

# Fitting the RNN to training set
regressor.fit(X_train, Y_train, batch_size = 32, epochs = 200)

# Saving the model
regressor.save('TrainedModel.model')




# Part 3 - Making the predictions and visualising the results

# Getting the real stock price of 2017
dataset_test = pd.read_csv('Google_Stock_Price_Test.csv')
real_stock_price = dataset_test.iloc[:, 1:2].values

# Getting the predicted stock price
inputs = real_stock_price
inputs = sc.transform(inputs)

# Reshaping inputs
inputs = np.reshape(inputs, (20, 1, 1))

￼
# Predicting
predicted_stock_prices = regressor.predict(inputs)
predicted_stock_prices = sc.inverse_transform(predicted_stock_prices)

# Visualizing the results
plt.plot(real_stock_price, color = 'red', label = 'Real Google Stock Price')
plt.plot(predicted_stock_prices, color = 'blue', label = 'Predicted Google Stock Price')
plt.title('Google Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Google Stock Price')
plt.legend
plt.show()




# Part 4 - Evaluating the RNN

import math
from sklearn.metrics import mean_squared_error
sp = pd.DataFrame(real_stock_price)
sp = sp.describe()
rmse = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_prices))



