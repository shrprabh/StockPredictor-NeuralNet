import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load your dataset
data = pd.read_csv('synthetic_apple_stock_data.csv')
# Convert the date column to datetime and sort data
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Select the 'Close' column for prediction
close_prices = data['Close'].values.reshape(-1, 1)

# Scale the data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(close_prices)

# Create training and testing datasets
train_size = int(len(scaled_data) * 0.8)
train_data = scaled_data[:train_size]
test_data = scaled_data[train_size:]
def create_dataset(data, time_step=3):
    X, Y = [], []
    for i in range(len(data) - time_step):
        a = data[i:(i + time_step), 0]
        X.append(a)
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)

time_step = 3
X_train, y_train = create_dataset(train_data, time_step)
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(3, 1)))
model.add(Dense(1))
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=1)


history = model.fit(X_train, y_train, epochs=100, batch_size=1)
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
plt.show()
# # Reshape input to be [samples, time steps, features]
# X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
# X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

# # Create the LSTM model
# model = Sequential()
# model.add(LSTM(units=50, return_sequences=True, input_shape=(time_step, 1)))
# model.add(LSTM(units=50, return_sequences=False))
# model.add(Dense(25))
# model.add(Dense(1))

# # Compile the model
# model.compile(optimizer='adam', loss='mean_squared_error')

# # Train the model
# model.fit(X_train, y_train, batch_size=32, epochs=100)

# # Predicting and inverse transformation
# train_predict = model.predict(X_train)
# test_predict = model.predict(X_test)

# # Inverse transform to get actual values
# train_predict = scaler.inverse_transform(train_predict)
# test_predict = scaler.inverse_transform(test_predict)

# # Plotting
# # Shift train predictions for plotting
# look_back = 100
# trainPredictPlot = np.empty_like(scaled_data)
# trainPredictPlot[:, :] = np.nan
# trainPredictPlot[look_back:len(train_predict)+look_back, :] = train_predict

# # Shift test predictions for plotting
# testPredictPlot = np.empty_like(scaled_data)
# testPredictPlot[:, :] = np.nan
# testPredictPlot[len(train_predict)+(look_back*2)+1:len(scaled_data)-1, :] = test_predict

# # Plot baseline and predictions
# plt.plot(scaler.inverse_transform(scaled_data))
# plt.plot(trainPredictPlot)
# plt.plot(testPredictPlot)
# plt.show()
