# dataset link: https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231
# Reference: https://www.kaggle.com/thebrownviking20/intro-to-recurrent-neural-networks-lstm-gru
# Videos:
# https://www.youtube.com/watch?v=ezom23o2boM
# https://www.youtube.com/watch?v=TYUBnCC18g0
# https://www.youtube.com/watch?v=Tj5wX5p4DCY

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')


# data Loading
data = pd.read_csv('AAPL_2006-01-01_to_2018-01-01.csv')
print(data.shape)
print(data.columns)
print(data.head())
print(data.tail(10))
print(data.info())
print(data.isnull().sum())
print(data.nunique())
print(data.describe(include = 'all'))


# Date Cleaning
## Convert date from string object to to datetime object
data['Date'] = pd.to_datetime(data['Date'])
data.index = data['Date']

## Drop column
data = data.drop(columns = ['Name'])
print(data.shape)
print(data.columns)
data.head()

# Choose 'High' attribute as prices
data = data[['High']]
print(data.shape)
print(data.columns)
data.head()


# Rename columns
# Let's give the column a human-friendly name: USD_CAD
data.rename(columns={'High': 'Apple Stock Price'}, inplace=True)
data.head()

# Visualization
data['Apple Stock Price'].plot()
data['Apple Stock Price'].plot(linewidth = 1, figsize = (15,5))
# https://finance.yahoo.com/quote/AAPL/chart?p=AAPL#eyJpbnRlcnZhbCI6Im1vbnRoIiwicGVyaW9kaWNpdHkiOjEsImNhbmRsZVdpZHRoIjo3LjgyNzU4NjIwNjg5NjU1Miwidm9sdW1lVW5kZXJsYXkiOnRydWUsImFkaiI6dHJ1ZSwiY3Jvc3NoYWlyIjp0cnVlLCJjaGFydFR5cGUiOiJsaW5lIiwiZXh0ZW5kZWQiOmZhbHNlLCJtYXJrZXRTZXNzaW9ucyI6e30sImFnZ3JlZ2F0aW9uVHlwZSI6Im9obGMiLCJjaGFydFNjYWxlIjoibGluZWFyIiwicGFuZWxzIjp7ImNoYXJ0Ijp7InBlcmNlbnQiOjEsImRpc3BsYXkiOiJBQVBMIiwiY2hhcnROYW1lIjoiY2hhcnQiLCJ0b3AiOjB9fSwibGluZVdpZHRoIjoyLCJzdHJpcGVkQmFja2dyb3VkIjp0cnVlLCJldmVudHMiOnRydWUsImNvbG9yIjoiIzAwODFmMiIsInN5bWJvbHMiOlt7InN5bWJvbCI6IkFBUEwiLCJzeW1ib2xPYmplY3QiOnsic3ltYm9sIjoiQUFQTCJ9LCJwZXJpb2RpY2l0eSI6MSwiaW50ZXJ2YWwiOiJtb250aCJ9XSwiY3VzdG9tUmFuZ2UiOnsic3RhcnQiOjExMzYxMzEyMDAwMDAsImVuZCI6MTUxNDgyMjQwMDAwMH0sImV2ZW50TWFwIjp7ImNvcnBvcmF0ZSI6eyJkaXZzIjp0cnVlLCJzcGxpdHMiOnRydWV9LCJzaWdEZXYiOnt9fSwicmFuZ2UiOnsiZHRMZWZ0IjoiMjAwNi0wMS0wMVQxNjowMDowMC4wMDBaIiwiZHRSaWdodCI6IjIwMTgtMDEtMzBUMTY6MDA6MDAuMDAwWiIsInBlcmlvZGljaXR5Ijp7ImludGVydmFsIjoibW9udGgiLCJwZXJpb2QiOjF9LCJwYWRkaW5nIjowfSwic3R1ZGllcyI6eyJ2b2wgdW5kciI6eyJ0eXBlIjoidm9sIHVuZHIiLCJpbnB1dHMiOnsiaWQiOiJ2b2wgdW5kciIsImRpc3BsYXkiOiJ2b2wgdW5kciJ9LCJvdXRwdXRzIjp7IlVwIFZvbHVtZSI6IiMwMGIwNjEiLCJEb3duIFZvbHVtZSI6IiNGRjMzM0EifSwicGFuZWwiOiJjaGFydCIsInBhcmFtZXRlcnMiOnsiaGVpZ2h0UGVyY2VudGFnZSI6MC4yNSwid2lkdGhGYWN0b3IiOjAuNDUsImNoYXJ0TmFtZSI6ImNoYXJ0In19fX0%3D

# Predict 2017 price with previous 11 (2006 - 2016) years of stock price
print(data.describe(include = 'all'))
train = data[:'2016']
test = data['2017':]

train["Apple Stock Price"].plot(figsize = (15,5), linewidth = 1, legend = True)
test["Apple Stock Price"].plot(figsize = (15,5), linewidth = 1, legend = True)
plt.legend(['Training set (2006 - 2016)','Test set (2017)'])
plt.title('Apple Stock Price')
plt.show()

# Feature Engineering
train['Apple Stock Price (-1)']= train['Apple Stock Price'].shift(1)
train.head()
for i in range(1, 50):
    train['Apple Stock Price (-' + str(i) + ')']= train['Apple Stock Price'].shift(i)
    test['Apple Stock Price (-' + str(i) + ')']= test['Apple Stock Price'].shift(i)
train.head(10)

# Drop rows with na
train = train.dropna()
test = test.dropna()
print(train.head())

print(train.iloc[:, 1:].head())
train_x = train.iloc[:, 1:].values
train_y = train['Apple Stock Price'].values
test_x = test.iloc[:, 1:].values
test_y = test['Apple Stock Price'].values

# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
print('Trainset Min = ' + str(train_x.min()))
print('Trainset Max = ' + str(train_x.max()))
print('Testset Min = ' + str(test_x.min()))
print('Testset Max = ' + str(test_x.max()))


# Modeling
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_x, train_y)

# Check accuracy
from sklearn.metrics import mean_squared_error
test['Apple Stock Price (Predict)'] = lr.predict(test_x)
print((mean_squared_error(test['Apple Stock Price (Predict)'], test['Apple Stock Price']))**0.5)
test['Apple Stock Price'].describe()

test["Apple Stock Price (Predict)"].plot(figsize = (15,5), linewidth = 1, legend = True)
test["Apple Stock Price"].plot(figsize = (15,5), linewidth = 1, legend = True)
plt.legend(['Apple Stock Price (Predict)','Apple Stock Price (Real)'])
plt.title('Apple Stock Price')
plt.show()

# Ways to improve the model
# More data - Longer duration, More detailed data (weekly or daily), More data sources
# Try out more models - svm, decision trees, random forest, neural network (LSTM)
# Tune the model
