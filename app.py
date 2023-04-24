import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pandas_datareader as pdr
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

st.title('Stock Trend Predictor')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

key = "ad55ba1b30b4faddd980f298a21bd730c0fe0bf4"
df = pdr.get_data_tiingo(user_input, api_key=key)

st.subheader('Date from 2018 - Present')
st.write(df.describe())

# st.subheader('Closing Price X Time Chart')
# fig = plt.figure(figsize= (12,6))
# plt.plot(df.close)
# st.pyplot(fig)

# st.subheader('Closing Price X Time Chart with 100-days moving average')
# mov_avg_100d = df.close.rolling(100).mean()
# fig = plt.figure(figsize= (12,6))
# plt.plot(mov_avg_100d,'g')
# plt.plot(df.close, 'r')
# st.pyplot(fig)

# st.subheader("Closing Price X Time Chart with 100 & 200-days moving average")
# mov_avg_100d = df.close.rolling(100).mean()
# mov_avg_200d = df.close.rolling(200).mean()
# fig = plt.figure(figsize= (12,6))
# plt.plot(mov_avg_100d)
# plt.plot(mov_avg_200d)
# plt.plot(df.close)
# st.pyplot(fig)

data_training = pd.DataFrame(df["close"][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df["close"][int(len(df)*0.70): int(len(df))])

scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)



model = load_model('spp_model.h5')

past_100d = data_training.tail(100)
final_df = past_100d.append(data_training, ignore_index=True)
input_data = scaler.fit_transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
  x_test.append(input_data [i-100: i])
  y_test.append(input_data [i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)
y_pred = model.predict(x_test)
scaler = scaler.scale_

scale_fact = 1/scaler[0]
y_pred = y_pred * scale_fact
y_test = y_test * scale_fact



st.subheader('Predictions X Actual')
fig = plt.figure(figsize=(12,6))
plt.plot(y_test, 'b', label = 'Actual Price')
plt.plot(y_pred, 'r', label = "Predicted Price")
plt.xlabel('Time(Days)')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig)