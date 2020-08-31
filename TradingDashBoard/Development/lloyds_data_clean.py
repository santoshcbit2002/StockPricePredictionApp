# Data Handling
import pandas as pd
import numpy as np
import datetime as dt
import warnings
import os
from numpy.random import seed

# Data Gathering
import requests
from bs4 import BeautifulSoup

# Modelling
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler


# Technical Analysis
import ta 
from talib import RSI, STOCH, WMA, MOM,MACD,ADOSC,WILLR,CCI

# Define Helper functions
def null_value_checks(data_frame):
    print('=====================================')
    print('====  Running Null Value Check  ==== ')
    temp=data_frame.isnull().sum(axis=0).sort_values(ascending=False).rename('Null_Values').reset_index()
    if len(temp[temp['Null_Values']>0])==0:
      print(' == There are no NaNs in the Data Frame ==')
    else:
      print(' == There are NaNs in the Data Frame. Columns are listed below ==')
      print(temp[temp['Null_Values']>0])
    print('=====================================')

print('<<< Processing main data....')   
# Read data into Dataframe
data=pd.read_csv('./Data/LLOY_HistoricPrices.csv')
print(data.head(2))
print('===================================================')
print(data.tail(2))
print(data.dtypes)

null_value_checks(data)

# Change Date format to datetime object
# Change Volume to int
# Reverse the dataframe so the data is increasing chronological from top to bottom. 
data['Date']=pd.to_datetime(data['Date'],format='%d/%m/%Y')
data['Volume']=data['Volume'].str.replace(',','').astype(int)
data=data[::-1]
print(data.dtypes)
print('===================================================')
print(data.head(2))
print('===================================================')
print(data.tail(2))
print('===================================================')
print(' Oldest time stamp : ' + str(data.Date.min()) + 'Latest time stamp: ' + str(data.Date.max()))
data_time_index=data.set_index('Date')
print(data_time_index.head(2))

# Delete those rows that have key variables 0s.
data_time_index=data_time_index[~((data_time_index['High']==0) |
               (data_time_index['Open']==0)  |
                (data_time_index['Close']==0) |
                (data_time_index['Low']==0) |
                (data_time_index['Volume']==0))]

print('<<< Applying Corrections....')                        
# Read corrections data into Dataframe
corrections=pd.read_excel('./Data/Data_Corrections.xlsx')
null_value_checks(corrections)
print(corrections.head(2))
corrections=corrections.set_index('Date')
corrected_data=pd.merge(data_time_index,corrections,on='Date',how='left')
corrected_data['Open']=corrected_data.apply(lambda x: x.Open_x if pd.isna(x.Open_y) else x.Open_y, axis=1)
corrected_data['High']=corrected_data.apply(lambda x: x.High_x if pd.isna(x.High_y) else x.High_y, axis=1)
corrected_data['Low']=corrected_data.apply(lambda x: x.Low_x if pd.isna(x.Low_y) else x.Low_y, axis=1)
corrected_data['Close']=corrected_data.apply(lambda x: x.Close_x if pd.isna(x.Close_y) else x.Close_y, axis=1)
corrected_data['Volume']=corrected_data.apply(lambda x: x.Volume_x if pd.isna(x.Volume_y) else x.Volume_y, axis=1)
corrected_data=corrected_data[['Open','High','Low','Close','Volume']]
corrected_data=corrected_data[~((corrected_data['High']==0) |
               (corrected_data['Open']==0)  |
                (corrected_data['Close']==0) |
                (corrected_data['Low']==0) |
                (corrected_data['Volume']==0))]
print(corrected_data.shape)

print('<<< Extract Technical Indicators....')  
corrected_data['SMA_10']=corrected_data['Close'].rolling(window=10).mean()
corrected_data['WMA_10']=WMA(corrected_data['Close'], timeperiod=10)
corrected_data['rsi']=RSI(corrected_data['Close'], timeperiod=10)
corrected_data['stoc_k'], corrected_data['stoc_d'] = STOCH(corrected_data['High'],corrected_data['Low'],corrected_data['Close'],10,6,0,6)
corrected_data['mom']=MOM(corrected_data['Close'], timeperiod=10)
corrected_data['macd'],_,_=MACD(corrected_data['Close'], fastperiod=12, slowperiod=26, signalperiod=10)
corrected_data['adosc']=ADOSC(corrected_data['High'], corrected_data['Low'], corrected_data['Close'], corrected_data['Volume'], fastperiod=3, slowperiod=10)
corrected_data['cci']=CCI(corrected_data['High'], corrected_data['Low'], corrected_data['Close'],timeperiod=10)
corrected_data['willr']=WILLR(corrected_data['High'], corrected_data['Low'], corrected_data['Close'],timeperiod=10)
null_value_checks(corrected_data)
corrected_data=corrected_data[~corrected_data['macd'].isnull()]
null_value_checks(corrected_data)
corrected_data_ta=corrected_data

print('<<< Completed extraction. Reading Forex Data....')  
exchange_rate=pd.read_csv('./Data/GBP_USD.csv')
print(exchange_rate.head(2))
exchange_rate['Match_Date']=pd.to_datetime(exchange_rate['Date'],format='%b %d, %Y')
print('Shape of Exchange Rate: ', exchange_rate.shape)
null_value_checks(exchange_rate)
exchange_rate=exchange_rate[::-1]

exchange_rate['Change %']=exchange_rate['Change %'].str.replace('%','')
exchange_rate['Change %']=exchange_rate['Change %'].astype(float)
exchange_rate.drop(columns=['Date','Change %'],inplace=True)

corrected_data_ta=corrected_data_ta.reset_index()
corrected_data_ta=pd.merge(corrected_data_ta,exchange_rate, left_on='Date', right_on='Match_Date',how='left')
corrected_data_ta.rename(columns={'Open_x': 'Lloyds_Open',
                              'High_x': 'Lloyds_High',
                              'Low_x': 'Lloyds_Low',
                              'Close':'Lloyds_Close',
                              'Price': 'GBP/USD Price',
                              'Open_y' : 'GBP/USD_Open_Price',
                              'High_y': 'GBP/USD_High_Price',
                              'Low_y': 'GBP/USD_Low_Price'
                             },inplace=True)
corrected_data_ta=corrected_data_ta[~corrected_data_ta['GBP/USD Price'].isnull()]
corrected_data_ta.drop(columns='Match_Date',inplace=True)
print(corrected_data_ta.shape)
null_value_checks(corrected_data_ta)
print('<<< Attached Forex. Reading Oil Data....')  

oil=pd.read_csv('./Data/Oil_HistoricPrices.csv')

oil['Vol.']=oil['Vol.'].apply(lambda x: x if x!='-' else '0K')
oil['Vol.']=oil['Vol.'].str.replace('K','')
oil['Vol.']=oil['Vol.'].str.replace('M','')
oil['Vol.']=oil['Vol.'].astype(float)
oil['Vol.']=oil['Vol.']*1000

oil.rename(columns={'Open': 'Oil_Open',
                    'High': 'Oil_High',
                    'Low': 'Oil_Low',
                    'Vol.':'Oil_Volume',
                    'Price': 'Oil_Price'},inplace=True)

oil['Match_Date']=pd.to_datetime(oil['Date'],format='%b %d, %Y')
oil=oil[::-1]
print(oil.shape)
null_value_checks(oil)

oil=oil.drop(columns=['Change %','Date'])

corrected_data_ta=pd.merge(corrected_data_ta,oil,left_on='Date',right_on='Match_Date',how='left')
null_value_checks(corrected_data_ta)
corrected_data_ta=corrected_data_ta[~corrected_data_ta['Oil_Price'].isnull()]
corrected_data_ta.drop(columns=['Match_Date'],inplace=True)
print(corrected_data_ta.shape)
corrected_data_ta.head(2)
print('<<< Attached Oil Data. Reading FTSE Data....')  

ftse=pd.read_csv('./Data/FTSE_HistoricalData.csv')

ftse['Date']=pd.to_datetime(ftse['Date'],format='%b %d, %Y')
ftse=ftse[['Date','Price','Open','High','Low']]
ftse['Price']=ftse['Price'].str.replace(',','').astype(float)
ftse['Open']=ftse['Open'].str.replace(',','').astype(float)
ftse['High']=ftse['High'].str.replace(',','').astype(float)
ftse['Low']=ftse['Low'].str.replace(',','').astype(float)
corrected_data_ta=pd.merge(corrected_data_ta,ftse,left_on='Date',right_on='Date',how='left')
null_value_checks(corrected_data_ta)
corrected_data_ta=corrected_data_ta[~corrected_data_ta['Price'].isnull()]
print(corrected_data_ta.shape)

corrected_data_ta=corrected_data_ta.rename(columns={"Price": "FTSE_Price", "Open": "FTSE_Open","High": "FTSE_High","Low": "FTSE_Low",})

corrected_data_ta['Weekday']=corrected_data_ta['Date'].apply( lambda x : x.weekday()).astype(str)
corrected_data_ta['Month']=corrected_data_ta['Date'].apply( lambda x : x.month)
corrected_data_ta['Year']=corrected_data_ta['Date'].apply( lambda x : x.year)
null_value_checks(corrected_data_ta)

print('<<< Completed Processing. Saving file....')  
corrected_data_ta.to_csv('./Data/Cleaned_data_lloyds.csv',encoding='utf-8',index=False)
