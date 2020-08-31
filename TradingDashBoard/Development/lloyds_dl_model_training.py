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
# Set a seed
seed(42)

# Read data into Dataframe
data=pd.read_csv('./Data/Cleaned_data_lloyds.csv')
print('Cleaned data shape: ',data.shape)

#print('\n ---- Null Values ---- ')
#print(data.isnull().sum(axis=0).sort_values(ascending=False))

print('Shape of deep learning data : ', data.shape)

#data.drop(columns=['Year','Date'],inplace=True)


# Split the data to train , test and val
data_train, data_val = train_test_split(data, train_size=0.90, test_size=0.10, shuffle=False)
print('Shape of Training data: ',data_train.shape)

val_count=int(data_val.shape[0]*0.8)
test_count=data_val.shape[0]-val_count
print(val_count)
print(test_count)

size_approx=data_val.shape[0]%2

data_test=data_val[val_count:][:]
data_val=data_val[:val_count][:]

print('Shape of Validation data: ',data_val.shape)
print('Shape of Test data: ',data_test.shape)

#print('Validation data: \n',data_val.tail(2))
#print('Test data: \n',data_test.tail(2))

dates_dump=np.save('./Predictions/dates_dump.npy',data_test['Date'].values)

#print('Training Data Sample :  '+'\n'+ '='*30 + '\n', data_train.head())
data_test.drop(columns=['Year','Date'],inplace=True)
data_val.drop(columns=['Year','Date'],inplace=True)
data_train.drop(columns=['Year','Date'],inplace=True)

# Scale the features
scaler = MinMaxScaler()
train_scaled=scaler.fit_transform(data_train)
test_scaled=scaler.transform(data_test)
val_scaled=scaler.transform(data_val)
print('Scaled Data Sample :  '+'\n'+ '='*30 + '\n', train_scaled)
print('Scaled Data Shape :  ', train_scaled.shape)
scaled_frame=pd.DataFrame(train_scaled,columns=data_train.columns)

time_steps=10
batch_size=10
num_of_epochs=500

# Define helper functions
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

results=pd.DataFrame()

print('<<< Training LSTM Model....')   

# Train LSTM Model
time_steps=10
batch_size=10
num_of_epochs=500

x_train,y_train=data_tensor(batch_size,time_steps,train_scaled,3)
x_test,y_test=data_tensor(batch_size,time_steps,test_scaled,3)
x_val,y_val=data_tensor(batch_size,time_steps,val_scaled,3)

lr_rate=0.001
optimisers = optimizers.RMSprop(lr=lr_rate)
filename='./Logs/'

def define_lstm_model():
    model = Sequential()
    model.add(LSTM(64, 
                    batch_input_shape=(batch_size,time_steps,x_train.shape[2]),
                    return_sequences=True,
                    kernel_initializer='random_uniform',
                    kernel_regularizer=regularizers.l2(0.0008)))
    model.add(Activation('relu'))
    model.add(Dropout(0.051))
    
    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='linear'))
    
    model.compile(loss='mean_absolute_error', optimizer=optimisers)
    
    model.summary()
    
    return model


start_time=dt.datetime.now()
model=define_lstm_model()

ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=40, min_delta=0)
    

lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, 
                        verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

csv_log = CSVLogger(os.path.join(filename,"log_lstm.csv"),separator=';',append=False)

MC = ModelCheckpoint(os.path.join(filename,"best_model_lstm.h5"), monitor='val_loss', verbose=0,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)
    
history = model.fit(x_train,y_train, epochs=num_of_epochs, batch_size=batch_size,verbose=0,
                    shuffle=False, validation_data=(x_val,y_val),callbacks=[lr_on_plateau,MC,csv_log])
print('** LSTM Training Time  : ',dt.datetime.now()-start_time)

print('<<< Training of LSTM Model complete....')   

# Visualize predictions

best_model = load_model(os.path.join(filename,'best_model_lstm.h5'))

y_pred = best_model.predict(x_test,batch_size=batch_size)
#y_pred = y_pred.flatten()

error = mean_absolute_error(y_test, y_pred)

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


y_pred_inv = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
y_test_inv = (y_test * scaler.data_range_[3]) + scaler.data_min_[3]
#print(y_pred_inv[0:15])
#print(y_test_inv[0:15])

np.save('./Predictions/LSTM_pred_dumps.npy',y_pred_inv)
np.save('./Predictions/LSTM_test_dumps.npy',y_test_inv)

results['lstm_loss']=history.history['loss']
results['lstm_val_loss']=history.history['val_loss']

lstm_preds=y_pred_inv

print('Average Training Mean Absolute Error : {:.4f}'.format(results['lstm_loss'].mean()))
print('<<< LSTM Predictions saved ....')   

#C-LSTM
print('<<< Training of C-LSTM Model started....')   

lr_rate=0.0002
time_steps=10
batch_size=10
num_of_epochs=500

optimisers = optimizers.Adam(lr=lr_rate)
x_train,y_train=data_tensor(batch_size,time_steps,train_scaled,3)
x_test,y_test=data_tensor(batch_size,time_steps,test_scaled,3)
x_val,y_val=data_tensor(batch_size,time_steps,val_scaled,3)

def define_cnn_lstm_model():
    model = Sequential()
    
    model.add(Conv1D(64, 3, activation='relu', input_shape=(time_steps,x_train.shape[2]),
                     kernel_initializer='random_uniform'))
    model.add(MaxPooling1D(pool_size=1))

    
    model.add(LSTM(128,return_sequences=True))
    model.add(Dropout(0.17))

    model.add(Flatten())
    model.add(Dense(128,activation='relu'))
    model.add(Dense(1,activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer=optimisers)
    
    model.summary()
    
    return model

## Define call Backs 
model=define_cnn_lstm_model()

ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=40, min_delta=0)
    

lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, 
                        verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

csv_log = CSVLogger(os.path.join(filename,"log_cnn_lstm.csv"),separator=';',append=False)


MC = ModelCheckpoint(os.path.join(filename,
                          "best_model_cnn_lstm.h5"), monitor='val_loss', verbose=0,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)
    
history = model.fit(x_train,y_train, epochs=num_of_epochs, batch_size=batch_size,verbose=0,
                    shuffle=False, validation_data=(x_val,y_val),callbacks=[lr_on_plateau,MC,csv_log])
# Visualize predictions

print('<<< Training of C-LSTM Model complete....')   
best_model = load_model(os.path.join(filename,'best_model_cnn_lstm.h5'))


y_pred = best_model.predict(x_test,batch_size=batch_size)
#y_pred = y_pred.flatten()
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




y_pred_inv = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
y_test_inv = (y_test * scaler.data_range_[3]) + scaler.data_min_[3]

np.save('./Predictions/C_LSTM_pred_dumps.npy',y_pred_inv)
np.save('./Predictions/C_LSTM_test_dumps.npy',y_test_inv)

results['cnn_lstm_loss']=history.history['loss']
results['cnn_lstm_val_loss']=history.history['val_loss']

cnn_lstm_preds=y_pred_inv

print('Average Training Mean Absolute Error : {:.4f}'.format(results['cnn_lstm_loss'].mean()))
print('<<< Predictions of C-LSTM Model complete....')   

print('<<< Training of GRU Model started....')   
# GRU
lr_rate=0.0005
time_steps=10
batch_size=10
num_of_epochs=500
optimisers = optimizers.Adam(lr=lr_rate)
x_train,y_train=data_tensor(batch_size,time_steps,train_scaled,3)
x_test,y_test=data_tensor(batch_size,time_steps,test_scaled,3)
x_val,y_val=data_tensor(batch_size,time_steps,val_scaled,3)

def define_gru_model():
    model = Sequential()
    model.add(GRU(128, 
                batch_input_shape=(batch_size,time_steps,x_train.shape[2]),
                return_sequences=True,
                kernel_initializer='random_uniform',
                kernel_regularizer=regularizers.l2(0.0008)))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))
    
    model.add(Flatten())
    model.add(Dense(256,activation='relu'))
    model.add(Dense(1,activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer=optimisers)
    
    model.summary()
    
    return model

## Define call Backs 
model=define_gru_model()

ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=40, min_delta=0)
    

lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, 
                        verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)


csv_log = CSVLogger(os.path.join(filename,"log_gru.csv"),separator=';',append=False)

MC = ModelCheckpoint(os.path.join(filename,
                          "best_model_gru.h5"), monitor='val_loss', verbose=0,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)
    
history = model.fit(x_train,y_train, epochs=num_of_epochs, batch_size=batch_size,verbose=0,
                    shuffle=False, validation_data=(x_val,y_val),callbacks=[lr_on_plateau,MC,csv_log])


print('<<< Training of GRU Model ended...')   
best_model = load_model(os.path.join(filename,'best_model_gru.h5'))


y_pred = best_model.predict(x_test,batch_size=batch_size)
#y_pred = y_pred.flatten()
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



y_pred_inv = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
y_test_inv = (y_test * scaler.data_range_[3]) + scaler.data_min_[3]


np.save('./Predictions/GRU_pred_dumps.npy',y_pred_inv)
np.save('./Predictions/GRU_test_dumps.npy',y_test_inv)

results['gru_loss']=history.history['loss']
results['gru_val_loss']=history.history['val_loss']
gru_preds=y_pred_inv

print('Average Training Mean Absolute Error : {:.4f}'.format(results['gru_loss'].mean()))
print('<<< Predictions of GRU Model complete....')   

print('<<< Training of CNN-1D Model started....')   
#CNN-1D
lr_rate=0.0002
time_steps=10
batch_size=10
num_of_epochs=500
optimisers = optimizers.Adam(lr=lr_rate)
x_train,y_train=data_tensor(batch_size,time_steps,train_scaled,3)
x_test,y_test=data_tensor(batch_size,time_steps,test_scaled,3)
x_val,y_val=data_tensor(batch_size,time_steps,val_scaled,3)
def define_cnn_model():
    model = Sequential()
    
    model.add(Conv1D(32, 3,input_shape=(time_steps,x_train.shape[2]),
                     kernel_initializer='random_uniform'))  
    model.add(Activation('relu'))
    #model.add(Conv1D(16, 3, activation='relu'))
    
    model.add(MaxPooling1D(pool_size=1))
    
    
    model.add(Flatten())
    model.add(Dense(64))
    model.add(Dense(1,activation='linear'))
    
    model.compile(loss='mean_squared_error', optimizer=optimisers)
    
    model.summary()
    
    return model
## Define call Backs 
model=define_cnn_model()


ES = EarlyStopping(monitor='val_loss', mode='min', verbose=0,patience=40, min_delta=0)
    

lr_on_plateau = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=40, 
                        verbose=0, mode='auto', min_delta=0, cooldown=0, min_lr=0)

csv_log = CSVLogger(os.path.join(filename,"log_cnn.csv"),separator=';',append=False)
                    

MC = ModelCheckpoint(os.path.join(filename,
                          "best_model_conv2d.h5"), monitor='val_loss', verbose=0,
                          save_best_only=True, save_weights_only=False, mode='min', period=1)
    
history = model.fit(x_train,y_train, epochs=num_of_epochs, batch_size=batch_size,verbose=0,
                    shuffle=False, validation_data=(x_val,y_val),callbacks=[lr_on_plateau,MC,csv_log])


best_model = load_model(os.path.join(filename,'best_model_conv2d.h5'))

print('<<< Training of CNN-1D Model ended....')   
y_pred = best_model.predict(x_test,batch_size=batch_size)
#y_pred = y_pred.flatten()
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


y_pred_inv = (y_pred * scaler.data_range_[3]) + scaler.data_min_[3]
y_test_inv = (y_test * scaler.data_range_[3]) + scaler.data_min_[3]

np.save('./Predictions/CNN2D_pred_dumps.npy',y_pred_inv)
np.save('./Predictions/CNN2D_test_dumps.npy',y_test_inv)


results['cnn_loss']=history.history['loss']
results['cnn_val_loss']=history.history['val_loss']

cnn_preds=y_pred_inv
print('Average Training Mean Absolute Error : {:.4f}'.format(results['cnn_loss'].mean()))

results.to_csv('./Logs/Results.csv')
"""
results[:100][['lstm_loss','cnn_lstm_loss','gru_loss','cnn_loss']].plot(figsize=(14,10))
plt.xlabel('Epochs')
plt.ylabel('Training loss')
plt.show()
preds=pd.DataFrame()
preds['real']=y_test_inv
preds['lstm']=lstm_preds
preds['gru']=gru_preds
preds['cnn_lstm']=cnn_lstm_preds
preds['cnn']=cnn_preds
plt.figure(figsize=(18,14))
plt.subplot(221)
plt.plot(preds['real'])
plt.plot(preds['lstm'])
plt.legend(['Real','lstm'])
plt.title('Real vs Prediction Price using LSTM Network')

plt.subplot(222)
plt.plot(preds['real'])
plt.plot(preds['gru'])
plt.legend(['real','gru'])
plt.title('Real vs Prediction Price using GRU Network')
          
plt.subplot(223)
plt.plot(preds['real'])
plt.plot(preds['cnn_lstm'])
plt.legend(['real','cnn_lstm'])
plt.title('Real vs Prediction Price using CNN-LSTM Network')
                    
plt.subplot(224)
plt.plot(preds['real'])
plt.plot(preds['cnn'])
plt.legend(['real','cnn'])
plt.title('Real vs Prediction Price using CNN-1D Network')
plt.show()
plt.figure(figsize=(18,14))
plt.subplot(221)
plt.plot(preds['real'][:5])
plt.plot(preds['lstm'][:5])
plt.legend(['Real','lstm'])
plt.title('Real vs Prediction Price using LSTM Network')

plt.subplot(222)
plt.plot(preds['real'][:5])
plt.plot(preds['gru'][:5])
plt.legend(['real','gru'])
plt.title('Real vs Prediction Price using GRU Network')
          
plt.subplot(223)
plt.plot(preds['real'][:5])
plt.plot(preds['cnn_lstm'][:5])
plt.legend(['real','cnn_lstm'])
plt.title('Real vs Prediction Price using CNN-LSTM Network')
                    
plt.subplot(224)
plt.plot(preds['real'][:5])
plt.plot(preds['cnn'][:5])
plt.legend(['real','cnn'])
plt.title('Real vs Prediction Price using CNN-1D Network')
plt.show()
"""