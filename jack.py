from myconf import *
import pandas as pd 
import quandl
import math
import numpy as np 
from sklearn import preprocessing, model_selection, svm
from sklearn.linear_model import LinearRegression


#load your quandl api key from conf file
quandl.ApiConfig.api_key = quandl_api

#this quandl code is different from the video tutorial(but the final number match), we need to use get_table instead, and add the ticker/date parameter
df = quandl.get_table('WIKI/PRICES', ticker='GOOGL', date = {'gte':'2004-8-19', 'lte':'2016-8-25'})
#print(df.head())

#only get the feature we need
df = df[['adj_open', 'adj_high', 'adj_low', 'adj_close',
 'adj_volume']]
#high & low percentage, percent of volatility
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0

#daily percent changes, the daily move
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0

#create a new data frame
df = df[['adj_close', 'HL_PCT', 'PCT_change', "adj_volume"]]

#define a label column
forecast_col = 'adj_close'
#deal with empty colum without deleting whole row
df.fillna(-999999, inplace = True)
forecast_out = int(math.ceil(0.01 * len(df)))
print(forecast_out)
#define a label
df['label'] = df[forecast_col].shift(-forecast_out)

#remove row with value as NaN
df.dropna(inplace = True)


#drop everything except label column and return as X
x = np.array(df.drop(['label'], 1))
y = np.array(df['label'])

#standardize the data
x = preprocessing.scale(x)
y = np.array(df['label'])

x_train, x_test, y_train, y_test = model_selection.train_test_split(x,y, test_size = 0.2)
clf = LinearRegression()
clf.fit(x_train, y_train)
accuracy = clf.score(x_test, y_test)
#test the accuracy of linear regression algorithm
print(accuracy)





















