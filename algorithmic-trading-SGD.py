# rateset link for daily exchange rate: https://data.gov.sg/dataset/exchange-rates-sgd-per-unit-of-usd-average-for-period-annual?resource_id=f927c39b-3b44-492e-8b54-174e775e0d98
# rateset link for Industrial product price index: https://data.gov.sg/dataset/consumer-price-index-monthly?resource_id=67d08d6b-2efa-4825-8bdb-667d23b7285e
# Reference: https://towardsratescience.com/rate-science-in-algorithmic-trading-d21a46d1565d

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# import cufflinks as cf
# cf.set_config_file(offline=True, world_readable=True, theme='ggplot')
plt.rcParams["figure.figsize"] = 12,8
plt.style.use('fivethirtyeight')


# data Loading
rate = pd.read_csv('exchange-rates.csv')
print(rate.shape)
print(rate.columns)
print(rate.head())
print(rate.tail(10))
print(rate.info())
print(rate.isnull().sum())
print(rate.nunique())
print(rate.describe(include = 'all'))

# cpi = pd.read_csv('consumer-price-index-division.csv')
cpi = pd.read_csv('consumer-price-index-group.csv')
print(cpi.shape)
print(cpi.columns)
print(cpi.head())
print(cpi.tail(10))
print(cpi.info())
print(cpi.isnull().sum())
print(cpi.nunique())
print(cpi.describe(include = 'all'))
print(cpi['level_1'].unique())
print(cpi['level_2'].unique())
print(cpi['level_3'].unique())
print(cpi.groupby(['level_2']).count())
print(cpi.groupby(['level_2', 'level_3'])['value'].count())
print(cpi['value'].unique())

# Date Cleaning
## Convert date from string object to to datetime object
rate['date'] = pd.to_datetime(rate['date'])
cpi['time'] = pd.to_datetime(cpi['month'])
cpi.drop(columns = ['month'], inplace = True)

## Drop column
cpi = cpi.drop(columns = ['level_1'])
print(cpi.shape)
print(cpi.columns)

## Investigate Missing Values
print(cpi[cpi['value'] == 'na'].head(10))
cpi[cpi['value'] == 'na'].groupby(['level_2', 'level_3'])['value'].count()

cpi[cpi['value'] == 'na'].describe()
cpi[cpi['value'] != 'na'].describe()

min = cpi[cpi['value'] != 'na']['time'].min()
cpi = cpi[cpi['time'] >= min]
cpi[cpi['value'] == 'na'].groupby(['level_2', 'level_3'])['value'].count()

## Drop rows
cpi = cpi[~cpi['level_3'].isin(['Catered Food','Other Miscellaneous Expenditure'])]
print(cpi[cpi['value'] == 'na'].head(10))
cpi[cpi['value'] == 'na'].groupby(['level_2', 'level_3'])['value'].count()
print(cpi.groupby(['level_2', 'level_3'])['value'].count())

# Merge Data
# Using this dataset has some serious limitations.
# First of all, it is only available monthly. That’s too slow for a real trading algorithm.
# Second, the factors in this index is all stuff about Canada, which is one half of the story or less when it comes to the USD/CAD trade.
# Third, we are not including data from the news, or other data sources.
# All this is fine, because we are just showing how it works in principle.
# In reality, there are lots of ways to leverage monthly macro data in daily trading, and to merge inputs from lots of types of data into a model.
# With these caveats in mind, here is how we merge the rates and CPI data into a usable format for our model:

# Get average monthly rate
rate['year'] = rate['date'].dt.year
rate['month'] = rate['date'].dt.month
print(rate.head(10))
rate = rate.groupby(['year','month'])['exchange_rate_usd'].mean().reset_index()
rate['day'] = 1
rate['time'] = pd.to_datetime(rate[['year', 'month', 'day']])
print(rate.head(10))

#
# cpi.pivot_table(values='value', index=cpi['month'], columns='level_2')
# data type error
cpi.info()
cpi['value'] = cpi['value'].astype(float)
cpi.info()
cpi = cpi.pivot_table(values = 'value', index = cpi['time'], columns = 'level_3')
# cpi.reset_index()
cpi.head()

# Merge two dataframes
combine = rate.merge(cpi, on = 'time')
print(combine.head(15))
print(combine.tail(10))
print(combine.describe(include = 'all'))
combine.head()

# Rename columns
# Let's give the column a human-friendly name: USD_CAD
combine.rename(columns={'exchange_rate_usd': 'USD_SGD'}, inplace=True)
combine.head()

## Drop column
combine.drop(columns = ['year', 'month', 'day'], inplace = True)
combine.head()


# Visualization
# combine.plot(kind='line', x='time', y='USD_SGD')
# combine.plot(kind='line', x='time', y='USD_SGD', linewidth=0.7)
# https://data.gov.sg/dataset/exchange-rates-sgd-per-unit-of-usd-average-for-period-annual?resource_id=f927c39b-3b44-492e-8b54-174e775e0d98

# combine['USD_SGD'].plot(linewidth=0.7)
# Force a common index with the economic rate we will be getting next
combine.index = combine['time']
combine.drop(columns = ['time'], inplace = True)
# combine['USD_SGD'].plot(linewidth=0.7)

# https://data.gov.sg/dataset/exchange-rates-sgd-per-unit-of-usd-average-for-period-annual?resource_id=2bc8eac9-8183-4e78-bf3e-9572a21a0ba4
# combine.iloc[-365:]['USD_SGD'].plot(linewidth=0.7)


combine.head()

# combine.plot(linewidth=2)
# combine.drop(columns = ['USD_SGD']).plot(linewidth=2)


# Feature Engineering
# A machine learning model has some input observations “x” and some output predictions “y” where the model is a function that makes the connection y=f(x).
# The model “f” maps from the observations to the predictions.
# In our case “x” is the CPI data, and we want to use it to predict price changes in USD_CAD, which is our “y” output.
# Before we can make predictions, we should dig into the correlations between CPI and USD_CAD to validate that these things are actually related to each other as we hypothesized.
# We will also look beyond correlations to see what correlations we see in future values of USD_CAD based upon past values in the CPI data.
# Put another way, we will look for indications that there are signals in CPI that we can use to make predictions about USD_CAD.

# Find relevant columns
# Tomorrow minus today's exchange rate gives the rate delta
# Intuition: When tomorrow's USD_CAD exchange rate is higher than today's, the result is positive
combine['dUSD_SGD']= combine['USD_SGD'].shift(-1) - combine['USD_SGD']
combine[['dUSD_SGD', 'USD_SGD']].head()
combine[['dUSD_SGD', 'USD_SGD']].tail()
combine = combine[~pd.isna(combine['dUSD_SGD'])]

corr = combine.corr()['dUSD_SGD'].sort_values(ascending=False).to_frame().reset_index()
print(corr)
corr = corr[corr['dUSD_SGD'] > 0.1]
print(corr)
categories_to_keep = list(corr['index'])
print(categories_to_keep)
categories_to_keep.remove('dUSD_SGD')

combine = combine[categories_to_keep + ['dUSD_SGD']]
print(combine.head())

# Create more features
# To make the predictions, we look back at the last 3 months of CPI indicators.
for category in categories_to_keep:
    combine[category + '(-1)'] = combine[category].shift(1)
    combine[category + '(-2)'] = combine[category].shift(2)

categories_to_keep
print(combine[['Accommodation', 'Accommodation(-1)', 'Accommodation(-2)']].head())

combine = combine.dropna()
print(combine[['Accommodation', 'Accommodation(-1)', 'Accommodation(-2)']].head())

# Scaling
y = combine['dUSD_SGD']
x = combine.drop(columns = ['dUSD_SGD'])

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
x[list(x)] = scaler.fit_transform(x[list(x)])
x.head()

# split train and test
from sklearn.model_selection import train_test_split
train_y, test_y, train_x, test_x = train_test_split(y, x, test_size = 0.3)

# Modeling
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_x, train_y)

# Check accuracy
from sklearn.metrics import mean_squared_error
print((mean_squared_error(lr.predict(test_x), test_y))**0.5)
y.max() - y.min()
print(y.describe())

from sklearn.linear_model import LogisticRegressionCV
lor = LogisticRegressionCV()
train_y = (train_y > 0).astype(int)
test_y  = (test_y > 0).astype(int)
train_y
lor.fit(train_x, train_y)

# Calculate accuracy score
from sklearn.metrics import accuracy_score
print(accuracy_score(lor.predict(test_x), test_y))

# Generate a report
pd.crosstab(pd.Series(test_y), lor.predict(test_x), rownames = ['True'], colnames = ['Pred'], margins = True)
# pd.crosstab(pd.Series(predict_real['real_signal']), pd.Series(predict_real['predict_signal']), rownames = ['True'], colnames = ['Pred'], margins = True)


# Ways to improve the model
# More data - Longer duration, More detailed data (weekly or daily), More data sources
# Try out more models - svm, decision trees, random forest, neural network
# Tune the model
#
