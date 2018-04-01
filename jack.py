import pandas as pd 
import quandl
from myconf import *
quandl.ApiConfig.api_key = quandl_api

#this quandl code is different from the video tutorial(but the final number match), we need to use get_table instead, and add the ticker/date parameter
df = quandl.get_table('WIKI/PRICES', ticker='GOOGL', date = {'gte':'2004-8-19', 'lte':'2004-8-25'})
#print(df.head())

#only get the feature we need
df = df[['adj_open', 'adj_low', 'adj_high', 'volume',
 'adj_low', 'adj_close', 'adj_volume']]
#high & low percentage, percent of volatility
df['HL_PCT'] = (df['adj_high'] - df['adj_close']) / df['adj_close'] * 100.0

#daily percent changes, the daily move
df['PCT_change'] = (df['adj_close'] - df['adj_open']) / df['adj_open'] * 100.0

#create a new data frame
df = df[['adj_close', 'HL_PCT', 'PCT_change', "adj_volume"]]

print(df.head())