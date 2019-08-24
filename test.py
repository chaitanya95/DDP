import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pylab import rcParams
rcParams['figure.figsize'] = 20,10
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler(feature_range=(0, 1))
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from pyramid.arima import auto_arima


df_complete_data = pd.read_csv("all_data.csv")
df_complete_data = df_complete_data.replace('?', 0)
print(df_complete_data.head())
df_complete_data['date_time']=df_complete_data['Date']+"|"+df_complete_data['Time']
print(df_complete_data.head())
df=pd.DataFrame()
df_complete_data['Date_new'] = pd.to_datetime(df_complete_data.date_time,format='%Y-%m-%d|%H:%M:%S')
print(df_complete_data.head())

df_complete_data.index = df_complete_data['Date_new']
data = df_complete_data.sort_index(ascending=True, axis=0)

new_data = pd.DataFrame()
new_data['Date'] = pd.to_numeric(df_complete_data['Date_new'])
new_data['Close'] = df_complete_data['Global_reactive_power']

#creating train and test sets
# dataset = new_data.values
# train = dataset[0:2105283,:]
# valid = dataset[2105283:,:]

train = data[:2105283]
valid = data[2105283:]
training = train['Global_reactive_power']
validation = valid['Global_reactive_power']

model = auto_arima(training, start_p=1, start_q=1,max_p=3, max_q=3, m=12,start_P=0, seasonal=True,d=1, D=1, trace=True,error_action='ignore',suppress_warnings=True)
model.fit(training)

forecast = model.predict(n_periods=248)
forecast = pd.DataFrame(forecast,index = valid.index,columns=['Prediction'])



# df_complete_data=df_complete_data.astype(str)
# plt.plot(df_complete_data['Global_reactive_power'], label='Global_active_power_history')
# plt.savefig('power.png')

# #converting dataset into x_train and y_train
# scaler = MinMaxScaler(feature_range=(0, 1))
# scaled_data = scaler.fit_transform(dataset)

# x_train, y_train = [], []
# for i in range(60,len(train)):
#     x_train.append(scaled_data[i-60:i,0])
#     y_train.append(scaled_data[i,0])
# x_train, y_train = np.array(x_train), np.array(y_train)

# x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# # create and fit the LSTM network
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
# model.add(LSTM(units=50))
# model.add(Dense(1))

# model.compile(loss='mean_squared_error', optimizer='adam')
# model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)

# #predicting 246 values, using past 60 from the train data
# inputs = new_data[len(new_data) - len(valid) - 60:].values
# inputs = inputs.reshape(-1,1)
# inputs  = scaler.transform(inputs)

# X_test = []
# for i in range(60,inputs.shape[0]):
#     X_test.append(inputs[i-60:i,0])
# X_test = np.array(X_test)

# X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
# closing_price = model.predict(X_test)
# closing_price = scaler.inverse_transform(closing_price)

# rms=np.sqrt(np.mean(np.power((valid-closing_price),2)))
# print(rms)

# train = new_data[:2105283]
# valid = new_data[2105283:]
# valid['Predictions'] = closing_price
# plt.plot(train['Close'])
# plt.plot(valid[['Close','Predictions']])
# plt.savefig('fitted.png')