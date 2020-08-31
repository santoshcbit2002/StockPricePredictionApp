# Data Handling
import pandas as pd
import numpy as np
import datetime as dt
import warnings
import os
from numpy.random import seed

# Visualization
from IPython.display import display
import matplotlib.pyplot as plt
import seaborn as sns

#PCA
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA


# Load specific forecasting tools 
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from keras.preprocessing.sequence import TimeseriesGenerator
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, Callback, History, EarlyStopping

from keras.models import Sequential, Model , load_model
from keras.layers import Dense, LSTM, GRU, Flatten, BatchNormalization, Activation, Dropout, TimeDistributed
from keras.layers.convolutional import Conv1D, MaxPooling1D,Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, CSVLogger, Callback, History, EarlyStopping
from keras import optimizers, regularizers

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error,mean_absolute_error


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
# Program Start
print('*'*50)
print('Program has started  - {}'.format(dt.datetime.now()))
print('Current Working Directory: {}'.format(os.getcwd()))
print('Loading Datasets : ')
print('*'*50)
seed(42)
# Read data into Dataframe
data=pd.read_csv('../Data/Cleaned_data_lloyds.csv')
# Sanity Check data 
print(data.shape)
print('\n ---- Null Values ---- ')
print(data.isnull().sum(axis=0).sort_values(ascending=False))
print('Shape of deep learning data : ', data.shape)
cols=list(data)
# Drop unused data 
#data.drop(columns=['Year','Date'],inplace=True)
# Split train, test and validation data 
data_train, data_test = train_test_split(data, train_size=0.90, test_size=0.10, shuffle=False)
print('Shape of Training data: ',data_train.shape)
data_val=data_test[:136][:]
print('Shape of Validation data: ',data_val.shape)
data_test=data_test[136:][:]

print('Shape of Test data: ',data_test.shape)
print('Training Data Sample :  '+'\n'+ '='*30 + '\n', data_train.head())
# Scaling
scaler = MinMaxScaler()
train_scaled=scaler.fit_transform(data_train)
test_scaled=scaler.transform(data_test)
val_scaled=scaler.transform(data_val)
print('Scaled Data Sample :  '+'\n'+ '='*30 + '\n', train_scaled)
print('Scaled Data Shape :  ', train_scaled.shape)
scaled_frame=pd.DataFrame(train_scaled,columns=data_train.columns)
# Set Model constants
time_steps=10
batch_size=10
num_of_epochs=500
# define helper functions
def data_tensor(batch_size,time_steps,array,close_price_loc):
    
    num_of_datapoints = array.shape[0] - time_steps
    num_of_datapoints_lost = num_of_datapoints%batch_size
    
    x = np.zeros((num_of_datapoints, time_steps, array.shape[1]))
    y = np.zeros(num_of_datapoints,)
 
    for i in range(num_of_datapoints):
        x[i] = array[i:time_steps+i]
        y[i] = array[time_steps+i, close_price_loc]
        
    if num_of_datapoints_lost > 0:
        return x[:-num_of_datapoints_lost], y[:-num_of_datapoints_lost]
    else:
        return x, y

def mean_absolute_percentage_error(y_true, y_pred): 
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mape= np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    #print('Mean absolute Percentage Error is : ',mape)
    return mape

# Set up tensors
x_train,y_train=data_tensor(batch_size,time_steps,train_scaled,3)
x_test,y_test=data_tensor(batch_size,time_steps,test_scaled,3)
x_val,y_val=data_tensor(batch_size,time_steps,val_scaled,3)

results=pd.DataFrame()
print(x_train.shape[2])

filename='/Users/santosh/OneDrive - University of Edinburgh/MSc/LJMU/Project/Code/logs'
best_model = load_model('best_model_gru.h5')

y_pred = best_model.predict(x_test,batch_size=batch_size)
y_pred = y_pred.flatten()
error = mean_squared_error(y_test, y_pred)

mape_test_2=mean_absolute_percentage_error(y_test[:2],y_pred[:2])
mape_test_5=mean_absolute_percentage_error(y_test[:5],y_pred[:5])

mae_test_2=mean_absolute_error(y_test[:2],y_pred[:2])
mae_test_5=mean_absolute_error(y_test[:5],y_pred[:5])

print('-'*30)
print('MAPE of Test Data (Short-term prediction - 2 Days) : ', round(mape_test_2,3))
print('MAPE of Test Data (Long-term prediction - 5 Days) : ', round(mape_test_5,3))
print('-'*30)
print('MAE of Test Data (Short-term prediction -2 Days) : ', round(mae_test_2,3))
print('MAE of Test Data (Long-term prediction - 5 Days) : ', round(mae_test_5,3))
print('-'*30)


#print("Mean Squared Error is : ", round(error,5))
#print(y_pred[0:15])
#print(y_test[0:15])

y_pred_inv = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
y_test_inv = (y_test * scaler.data_range_[3]) + scaler.data_min_[3]
#print(y_pred_inv[0:15])
#print(y_test_inv[0:15])


plt.figure(figsize=(14, 8), dpi=70)
plt.plot(y_pred_inv)
plt.plot(y_test_inv)
plt.title('Prediction vs Real Stock Price for length of Test Set')
plt.ylabel('Price')
plt.xlabel('Days')
plt.legend(['Prediction', 'Real'], loc='upper left')
plt.show()
mae_test_1=mean_absolute_error(y_test[:1],y_pred[:1])
mae_test_7=mean_absolute_error(y_test[:7],y_pred[:7])
mape_test_1=mean_absolute_percentage_error(y_test[:1],y_pred[:1])
rmse_test_1=np.sqrt(mean_squared_error(y_test[:2],y_pred[:2]))
print('MAPE of Test Data (1 day prediction) : ', round(mape_test_1,3))
print('MAE of Test Data (1 day prediction) : ', round(mae_test_1,3))
print('RMSE of Test Data (1 day prediction) : ', round(rmse_test_1,3))
print('MAE of Test Data (7 day prediction) : ', round(mae_test_7,3))
np.save('lloyds_predicted.npy',y_pred_inv)
np.save('lloyds_test.npy',y_pred_inv)
