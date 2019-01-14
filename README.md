# NTUOSS Data Science for Algorithmic Trading Workshop

*by [Tu Anqi](https://github.com/anqitu) for NTU Open Source Society*

This workshop assumes basic knowledge of Python.

**Disclaimer:** *This document is only meant to serve as a reference for the attendees of the workshop. It does not cover all the concepts or implementation details discussed during the actual workshop.*
___

### Workshop Details
**When**: Friday, 5 April 2018. 6:30 PM - 8:30 PM.</br>
**Where**: LT1 </br>
**Who**: NTU Open Source Society

### Questions
Please raise your hand any time during the workshop or email your questions to [me](mailto:anqitu@outlook.com) later.

### Errors
For errors, typos or suggestions, please do not hesitate to [post an issue](https://github.com/anqitu/NTUOSS-DataScienceforAlgorithmicTradingWorkshop/issues/new). Pull requests are very welcome! Thanks!
___

## Task 0 - Getting Started

#### 0.1 Introduction

For this tutorial, we'll be training an algorithmic trading model to predict Apple's stock market price on Colaboratory.


<p align="center">
  <img src="Images/what-is-algorithm-trading.png" width="500">
</p>

1. What is [Algorithmic Trading](https://www.investopedia.com/articles/active-trading/101014/basics-algorithmic-trading-concepts-and-examples.asp)?\
Algorithmic trading (automated trading, black-box trading or simply algo-trading) is the process of using computers programed to follow a defined set of instructions (an algorithm) for placing a trade in order to generate profits at a speed and frequency that is impossible for a human trader. The defined sets of rules are based on timing, price, quantity or any mathematical model. Apart from profit opportunities for the trader, algo-trading makes markets more liquid and makes trading more systematic by ruling out the impact of human emotions on trading activities.</br>
**Video Links:**
- https://www.youtube.com/watch?v=ezom23o2boM
- https://www.youtube.com/watch?v=TYUBnCC18g0
- https://www.youtube.com/watch?v=Tj5wX5p4DCY

2. What is [Colaboratory](https://colab.research.google.com/notebooks/welcome.ipynb)?\
Colaboratory is a Google research project created to help disseminate machine learning education and research. It is a free Jupyter notebook environment that requires no setup and runs entirely in a virtual machine (VM) hosted in the cloud.


#### 0.2 Overview

Here is an overview of today's workshop.
- Prepare Data
-

<p align="center">
  <img src="" width="700">
</p>

#### 0.3 Initial Setup

Open this google drive [folder](https://drive.google.com/open?id=1DC_FWf2LZSrOqcNb2qVCRP37AZRDrdGW) and add to your own google drive.

<p align="center">
  <img src="" width="500">
</p>


Then, download and unzip this [folder](https://workupload.com/file/h2Vcjr4R), and upload it to your google drive.

<p align="center">
  <img src="" width="700">
</p>

Now, open the 'NTUOSS-ImageRecognitionWorkshop' folder. You will see a 'start.ipynb' file, which is a Jupyter notebook that contains the incomplete script that you are going to code on for today's workshop.

Let's open the 'start.ipynb' file together to officially start the coding part of today's workshop: Right click 'start.ipynb' file -> Select 'Open with' -> Select 'Colaboratory'.

<p align="center">
  <img src="" width="500">
</p>

If you do not have any app to open the notebook yet, follow the steps as shown below: Right click 'start' file -> Select 'Connect more apps' -> Search for 'colaboratory' -> Click on 'connect'.

<p align="center">
  <img src="" width="500">
</p>

<p align="center">
  <img src="" width="500">
</p>


## Task 1 - Set Up

#### 1.1 Mount Google Drive

To import tha data into the VM, we will mount the google drive on the machine using `google-drive-ocamlfuse`. ([Reference](https://gist.github.com/Joshua1989/dc7e60aa487430ea704a8cb3f2c5d6a6))

```python
# Task: 1.2.1 Install google-drive-ocamlfuse
!apt-get install -y -qq software-properties-common python-software-properties module-init-tools
!add-apt-repository -y ppa:alessandro-strada/ppa 2>&1 > /dev/null
!apt-get update -qq 2>&1 > /dev/null
!apt-get -y install -qq google-drive-ocamlfuse fuse
```

> **Shell Assignment** <br>
The exclamation mark `!` (or bang) allows us to execute shell commands outside of the Python interpreter. This is useful for when you need to access the underlying shell, like installing dependencies, traversing directories or moving files around.

Then, authenticate and get credentials for your google drive.
```python
# Task: 1.2.2 Authenticate and get credentials
from google.colab import auth
auth.authenticate_user()
from oauth2client.client import GoogleCredentials
creds = GoogleCredentials.get_application_default()

import getpass
!google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret} < /dev/null 2>&1 | grep URL
vcode = getpass.getpass()
!echo {vcode} | google-drive-ocamlfuse -headless -id={creds.client_id} -secret={creds.client_secret}
```
You will be asked **twice** to authenticate the access to your drive. At each step a token will be generated:
- Click on the link to log into your google account.
- Allow access to your drive.
- Copy the token (The token looks like this - 4/PABmEY7BRPd3jPR9BI9I4R99gc9QITTYFUVDU76VR)
- Switch to the notebook to paste the token in the 'Enter verification code' bar.
- Press 'Enter'

And then you can mount your google drive in your current virtual machine.
```python
# TASK 1.2.3: Mount Google Drive in local Colab VM
!mkdir -p drive
!google-drive-ocamlfuse drive
!ls
```

You should see a /drive folder inside the current working directory.
```
adc.json  datalab  drive  sample_data
```

Running the code below should let you see the folders inside your google drive.
```python
!ls drive
```

Then, access our working directory /drive/NTUOSS-DataScienceforAlgorithmicTradingWorkshop by running the code below.
```python
!ls drive/NTUOSS-DataScienceforAlgorithmicTradingWorkshop
```

Lastly, check the data directory.
```python
!ls drive/NTUOSS-DataScienceforAlgorithmicTradingWorkshop
```

#### 1.2 Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
plt.style.use('fivethirtyeight')

#ignore warnings
import warnings
warnings.filterwarnings('ignore')
```


## Task 2 - Prepare Data
#### 2.1 Check Data
```python
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
```

#### 2.2 Clean Data
```python
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
```

#### 2.2 Clean Data
```python
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
```

## Task 3 - Visualiza Data (Exploratory Data Analysis)
```python
# Visualization
data['Apple Stock Price'].plot()
data['Apple Stock Price'].plot(linewidth = 1, figsize = (15,5))
```
[Yahoo Finance!](https://finance.yahoo.com/quote/AAPL/chart?p=AAPL#eyJpbnRlcnZhbCI6Im1vbnRoIiwicGVyaW9kaWNpdHkiOjEsImNhbmRsZVdpZHRoIjo3LjgyNzU4NjIwNjg5NjU1Miwidm9sdW1lVW5kZXJsYXkiOnRydWUsImFkaiI6dHJ1ZSwiY3Jvc3NoYWlyIjp0cnVlLCJjaGFydFR5cGUiOiJsaW5lIiwiZXh0ZW5kZWQiOmZhbHNlLCJtYXJrZXRTZXNzaW9ucyI6e30sImFnZ3JlZ2F0aW9uVHlwZSI6Im9obGMiLCJjaGFydFNjYWxlIjoibGluZWFyIiwicGFuZWxzIjp7ImNoYXJ0Ijp7InBlcmNlbnQiOjEsImRpc3BsYXkiOiJBQVBMIiwiY2hhcnROYW1lIjoiY2hhcnQiLCJ0b3AiOjB9fSwibGluZVdpZHRoIjoyLCJzdHJpcGVkQmFja2dyb3VkIjp0cnVlLCJldmVudHMiOnRydWUsImNvbG9yIjoiIzAwODFmMiIsInN5bWJvbHMiOlt7InN5bWJvbCI6IkFBUEwiLCJzeW1ib2xPYmplY3QiOnsic3ltYm9sIjoiQUFQTCJ9LCJwZXJpb2RpY2l0eSI6MSwiaW50ZXJ2YWwiOiJtb250aCJ9XSwiY3VzdG9tUmFuZ2UiOnsic3RhcnQiOjExMzYxMzEyMDAwMDAsImVuZCI6MTUxNDgyMjQwMDAwMH0sImV2ZW50TWFwIjp7ImNvcnBvcmF0ZSI6eyJkaXZzIjp0cnVlLCJzcGxpdHMiOnRydWV9LCJzaWdEZXYiOnt9fSwicmFuZ2UiOnsiZHRMZWZ0IjoiMjAwNi0wMS0wMVQxNjowMDowMC4wMDBaIiwiZHRSaWdodCI6IjIwMTgtMDEtMzBUMTY6MDA6MDAuMDAwWiIsInBlcmlvZGljaXR5Ijp7ImludGVydmFsIjoibW9udGgiLCJwZXJpb2QiOjF9LCJwYWRkaW5nIjowfSwic3R1ZGllcyI6eyJ2b2wgdW5kciI6eyJ0eXBlIjoidm9sIHVuZHIiLCJpbnB1dHMiOnsiaWQiOiJ2b2wgdW5kciIsImRpc3BsYXkiOiJ2b2wgdW5kciJ9LCJvdXRwdXRzIjp7IlVwIFZvbHVtZSI6IiMwMGIwNjEiLCJEb3duIFZvbHVtZSI6IiNGRjMzM0EifSwicGFuZWwiOiJjaGFydCIsInBhcmFtZXRlcnMiOnsiaGVpZ2h0UGVyY2VudGFnZSI6MC4yNSwid2lkdGhGYWN0b3IiOjAuNDUsImNoYXJ0TmFtZSI6ImNoYXJ0In19fX0%3D)

```python
# Predict 2017 price with previous 11 (2006 - 2016) years of stock price
print(data.describe(include = 'all'))
train = data[:'2016']
test = data['2017':]

train["Apple Stock Price"].plot(figsize = (15,5), linewidth = 1, legend = True)
test["Apple Stock Price"].plot(figsize = (15,5), linewidth = 1, legend = True)
plt.legend(['Training set (2006 - 2016)','Test set (2017)'])
plt.title('Apple Stock Price')
plt.show()
```

## Task 4 - Preprocess Data
#### 4.1 Feature Engineering
```python
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
```

```python

#### 4.2 Scaling
# Scaling
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
train_x = scaler.fit_transform(train_x)
test_x = scaler.transform(test_x)
print('Trainset Min = ' + str(train_x.min()))
print('Trainset Max = ' + str(train_x.max()))
print('Testset Min = ' + str(test_x.min()))
print('Testset Max = ' + str(test_x.max()))
```

## Task 5 - Train the Model

```python
from sklearn.linear_model import LinearRegression

lr = LinearRegression()
lr.fit(train_x, train_y)
```


## Task 5 - Test Model
As we can see from the training history, the final accuracy score is 74% for train data and 78% for validation data. Now let us test the model with our own test data.

```python
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
```

There are also many ways to improve this model, such as
- More data (Longer duration, More data sources)
- Try out more models - svm, decision trees, random forest, neural network (LSTM)
- Tune the model
___

## Acknowledgements

Many thanks to [clarencecastillo](https://github.com/clarencecastillo) for carefully testing this walkthrough and to everybody else in [NTU Open Source Society](https://github.com/ntuoss) committee for making this happen! :kissing_heart::kissing_heart::kissing_heart:

## Resources
[Kaggle Stock Market Dataset](https://www.kaggle.com/szrlee/stock-time-series-20050101-to-20171231)
