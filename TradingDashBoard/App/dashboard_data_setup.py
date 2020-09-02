import requests
import pandas as pd
import time
import datetime
import re
import os
import numpy as np

import pandas_market_calendars as mcal

print('*'*50)
print('Program has started  - {}'.format(datetime.datetime.now()))
print('Current Working Directory: {}'.format(os.getcwd()))



table_data=pd.read_csv('./Data/table_data.csv')
model_data=pd.read_csv('./Data/model_data.csv')
print(model_data.head(3))

#model_data['Date']=model_data['Date'].apply(lambda x: x.split('/')[2]+'-'+x.split('/')[1]+'-'+x.split('/')[0])

current_day=table_data[table_data['day']==1]

previous_day_record=[0]+current_day.values.tolist()[0][1:]
print(previous_day_record)
tickers=['LLOY.XLON','BARC.XLON','RBS.XLON','FTSE.INDX']
ticker_list=[1]

for i in tickers:

    params = {
        'access_key': '8231f84ca3b18528015548706c8bd76f',
        'symbols':i

        }
    response=requests.get('http://api.marketstack.com/v1/eod/latest', params)
    response=response.json()
    
    ticker_list.append(response['data'][0]['high'])
    ticker_list.append(response['data'][0]['open'])
    ticker_list.append(response['data'][0]['close'])
    ticker_list.append(response['data'][0]['low'])

    if i=='LLOY.XLON':
      lloyds_close=response['data'][0]['close']
      latest_date=response['data'][0]['date'][:10]


# Append the business date to the model_data and write data to Data folder
#year=latest_date.split('-')[0]
#month=latest_date.split('-')[1]
#day=latest_date.split('-')[2]

#latest_date_upd=day+'/'+month+'/'+year
#latest_date_upd_2=year+'-'+month+'-'+day

print('latest date: ',latest_date)
record_dict={'Date': latest_date,
              'Lloyds_Close': lloyds_close}

updated_model_data=model_data.append(record_dict,ignore_index=True)

# Extract next business day 
LSE = mcal.get_calendar('LSE')
a=LSE.valid_days(start_date=latest_date, end_date='2020-12-31')[:20]
next_business_date=a[1].strftime('%Y-%m-%d')
print('Next business day : {}'.format(next_business_date))

updated_model_data=updated_model_data[updated_model_data['Lloyds_Close']!=' ']

record_dict={'Date': next_business_date,
              'Lloyds_Close': ' '}

updated_model_data=updated_model_data.append(record_dict,ignore_index=True)
#updated_model_data.drop_duplicates(subset=['Date'],inplace=True)
updated_model_data.to_csv('./Data/model_data.csv',index=False)


# Now extract USD/GBP data

params = {
  'access_key': 'd55d57ff0d2c79990689e03ea82691d8',
  'date':latest_date
}
api_result=requests.get('http://api.currencylayer.com/historical', params)
response=api_result.json()

value=response['quotes']['USDGBP']
forex=np.round(1/float(value),3)

print(forex)

ticker_list.append(forex)

print(ticker_list)
print('*****')
print(previous_day_record)

cols=['day','LLOY_high','LLOY_open','LLOY_close','LLOY_low','BARC_high','BARC_open','BARC_close','BARC_low','RBS_high','RBS_open','RBS_close'	,'RBS_low','FTSE_high','FTSE_open','FTSE_close','FTSE_low','USDGBP','Oil']
updated_table_data=pd.DataFrame([ticker_list,previous_day_record],columns=cols)
updated_table_data.to_csv('./Data/table_data.csv',index=False)


# Get the latest business date 
print('Program has ended  - {}'.format(datetime.datetime.now()))