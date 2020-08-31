# Data Handling
import pandas as pd
import numpy as np
import datetime as dt
import warnings
import os
from numpy.random import seed

from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error

# Technical Analysis
import ta 

# Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns


from statsmodels.graphics.tsaplots import plot_acf, plot_pacf, acf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller,grangercausalitytests
from statsmodels.tsa.ar_model import AR, ARResults
from statsmodels.tsa.arima_model import ARMA,ARIMA,ARMAResults,ARIMAResults
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from pmdarima import auto_arima
import statsmodels.tsa.api as smt
from statsmodels.tsa.api import VAR

from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.statespace.tools import diff

seed(42)
# Read data into Dataframe

data=pd.read_csv('./Data/model_data.csv')


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
    
    
def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    #print('Mean absolute Percentage Error is : ',mape)
    return mape

def adf_test(series,title='',verbose=False):
    """
    Pass in a time series and an optional title, returns an ADF report
    """
    if verbose:
        print(f'Augmented Dickey-Fuller Test: {title}')
        print('===============================================')
    
    result = adfuller(series.dropna(),autolag='AIC') 
    
    labels = ['ADF test statistic','p-value','Lags','Data']
    out = pd.Series(result[0:4],index=labels)
    
    for key,val in result[4].items():
        out[f'critical value ({key})']=val
        
    if verbose:
        print(out.to_string())          
        print('===============================================')
        if result[1] <= 0.05:
            print("Strong evidence against the null hypothesis")
            print("Reject the null hypothesis")                
            print("Data has no unit root and is stationary")
        else:
            print("Weak evidence against the null hypothesis")
            print("Fail to reject the null hypothesis")
            print("Data has a unit root and is non-stationary")
    return out

def adf_report(dataframe):
    adf_results=pd.DataFrame()
    adf_report=pd.DataFrame()
    
    for col in list(dataframe):
        result=adf_test(data_adf_test[col],title=col,verbose=False)
        adf_results[col]=result.apply(lambda x : round(x,2))
        adf_report=adf_results.loc['p-value'].reset_index()
        
    print("==========================================================================")
    print("The following attributes have Strong evidence against the null hypothesis")
    print("Reject the null hypothesis")
    print("Data has no unit root and is Stationary"+'\n')
    print(adf_report[adf_report['p-value'] <= 0.05][['index','p-value']])  
    print('\n'+"==========================================================================")
    print("The following attributes have Weak evidence against the null hypothesis")
    print("Fail to Reject the null hypothesis")
    print("Data has a unit root and is non-stationary"+'\n')
    print(adf_report[adf_report['p-value'] > 0.05][['index','p-value']])  
    
    return adf_results

print(data.head(5))
data=data[:-1][:]
data['Lloyds_Close']=data['Lloyds_Close'].astype(float)
#_=adf_test(data['Lloyds_Close'],'Close Price of Lloyds Share',verbose=True)

data_train, data_test = train_test_split(data, train_size=0.90, test_size=0.10, shuffle=False)

print('Shape of Training data: ',data_train.shape)
print('Shape of Test data: ',data_test.shape)

arima_fits=auto_arima(data_train['Lloyds_Close'],start_p=0,max_p=7,start_q=0,max_q=7,seasonal=True,trace=True)

#print(arima_fits.summary())

def arima_model(train_data, test_data, order):
    print('Shape of train data : ',train_data.shape)
    print('Shape of test data : ',test_data.shape)
    print('Order of ARIMA : ',order)
    train_array =train_data['Lloyds_Close'].values
    test_array = test_data['Lloyds_Close'].values
    history = [x for x in train_array]
    predictions = []
    return_values = []
    for i in range(len(test_array)):
        model = ARIMA(history, order=order)
        result = model.fit()
        output = result.forecast()
        predictions.append(output[0])
        history.append(test_array[i])
    
    print('** Model fit Complete **')
    for j in range(len(predictions)):
        return_values.append(list(predictions[j])[0])
        
    return return_values
  
predictions_test=arima_model(data_train,data_test,(2,1,0))

np.save('./Data/development_predictions.npy',predictions_test)

data_all_but_10=data[:-10]

train_array =data_all_but_10['Lloyds_Close'].values

history = [x for x in train_array]
preds=[]
for i in range(0,11):
    model = ARIMA(history, order=(2,1,0))
    result = model.fit()
    output = result.forecast()
    #print(output[0])
    preds.append(output[0][0])
    history.append(output[0][0])

print(preds)
np.array(preds)
np.save('./Data/predictions_dump.npy',preds)

