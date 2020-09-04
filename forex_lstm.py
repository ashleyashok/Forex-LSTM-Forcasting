import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('E:/Python/Foreign Exchange Rates/Foreign_Exchange_Rates.csv')
df['Time Serie'] = pd.to_datetime(df['Time Serie'])
df.set_index('Time Serie', inplace = True)

df = df.loc[(df['AUSTRALIA - AUSTRALIAN DOLLAR/US$']!= 'ND') & (df['INDIA - INDIAN RUPEE/US$']!= 'ND')]
df = df.apply(pd.to_numeric, errors = 'ignore')
df = df.iloc[:, 1:]


tri = df['INDIA - INDIAN RUPEE/US$'].apply(lambda x: 1 if x == 'ND' else 0 )

corr = df.corr()

series = df['CANADA - CANADIAN DOLLAR/US$'].values
split_time = int(len(series) * 0.70)
series_train = series[:split_time]
series_valid = series[split_time:]

def window_method(arr, t):
    x = []
    y = []
    
    for i in range(t, arr.shape[0]):
        x.append(arr[i-t:i])
        label = 1 if arr[i] > arr[i-1] else 0
        y.append(label)
    
    return np.array(x).reshape(-1,t,1), np.array(y)

x_train, y_train = window_method(series_train, 30)
x_valid, y_valid = window_method(series_valid, 30)

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv1D(filters=32, kernel_size=5,
                      strides=1, padding="causal",
                      activation="relu",
                      input_shape=[None, 1]),
    tf.keras.layers.LSTM(128, return_sequences=True),
    tf.keras.layers.LSTM(64, return_sequences=True),
    tf.keras.layers.Dense(30, activation="relu"),
    tf.keras.layers.Dense(10, activation="relu"),
    tf.keras.layers.Dense(1, activation = 'sigmoid')    
    
    ])

model.compile(loss = 'binary_crossentropy', optimizer = 'sgd', metrics = ['accuracy'])

history = model.fit(x_train, y_train, validation_data = (x_valid, y_valid), epochs = 20, batch_size=32)

plt.plot(history.history['loss'])
