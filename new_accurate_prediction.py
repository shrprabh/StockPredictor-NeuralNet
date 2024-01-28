import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional
from ta import add_all_ta_features  # For technical indicators

# Load the dataset
data = pd.read_csv('INFY.NS-3.csv')

# Convert the date column to datetime and sort the data
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Add technical indicators
data = add_all_ta_features(data, open="Open", high="High", low="Low", close="Close", volume="Volume")

# Fill any NaNs or missing values
data = data.fillna(method='bfill')

# Convert prices to returns
data['Close'] = data['Close'].pct_change()
data['High'] = data['High'].pct_change()
data['Low'] = data['Low'].pct_change()
data.dropna(inplace=True)  # Drop rows with NaN values that result from pct_change()

# Initialize scalers for each feature
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_high = MinMaxScaler(feature_range=(0, 1))
scaler_low = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the returns
scaled_close = scaler_close.fit_transform(data[['Close']])
scaled_high = scaler_high.fit_transform(data[['High']])
scaled_low = scaler_low.fit_transform(data[['Low']])

# Define the time step for previous days' data to use for prediction
time_step = 3

# Function to create the dataset for LSTM
def create_dataset(X, high, low, time_step):
    Xs, Ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step)]
        Xs.append(v)
        Ys.append([high[i + time_step], low[i + time_step]])
    return np.array(Xs), np.array(Ys)

# Prepare the training data
X_train, y_train = create_dataset(scaled_close, scaled_high, scaled_low, time_step)

# Reshape the input to be [samples, time steps, features] for the LSTM
X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

# Create the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='tanh', return_sequences=True), input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100, activation='tanh')))
model.add(Dropout(0.3))
model.add(Dense(2))  # Predicting two values: High and Low
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prediction for the test dataset
train_size = int(len(scaled_close) * 0.8)
X_test, y_test = create_dataset(scaled_close[train_size:], scaled_high[train_size:], scaled_low[train_size:], time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
predicted_high_low_test = model.predict(X_test)

# Inverse transform to get actual values
predicted_high_test = scaler_high.inverse_transform(predicted_high_low_test[:, 0].reshape(-1, 1))
predicted_low_test = scaler_low.inverse_transform(predicted_high_low_test[:, 1].reshape(-1, 1))
actual_high_test = scaler_high.inverse_transform(y_test[:, 0].reshape(-1, 1))
actual_low_test = scaler_low.inverse_transform(y_test[:, 1].reshape(-1, 1))
last_sequence = scaled_close[-time_step:]

# Predict the next 10 days
future_dates = pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
predicted_highs = []
predicted_lows = []

for i in range(10):
    last_sequence_input = last_sequence.reshape((1, time_step, 1))
    predicted_high_low = model.predict(last_sequence_input)
    
    # Append the predictions
    predicted_highs.append(predicted_high_low[0, 0])
    predicted_lows.append(predicted_high_low[0, 1])
    
    # Update the last sequence
    new_point = np.array([[predicted_high_low[0, 0]]])  # Reshape new point to be [1, 1]
    last_sequence = np.append(last_sequence[1:], new_point, axis=0)  # Append new point

predicted_highs = scaler_high.inverse_transform(np.array(predicted_highs).reshape(-1, 1))
predicted_lows = scaler_low.inverse_transform(np.array(predicted_lows).reshape(-1, 1))

# Plotting the actual and predicted prices
plt.figure(figsize=(15, 7))
plt.plot(data['Date'][train_size + time_step:train_size + time_step + len(actual_high_test)], actual_high_test, label='Actual High', color='blue')
plt.plot(data['Date'][train_size + time_step:train_size + time_step + len(actual_low_test)], actual_low_test, label='Actual Low', color='orange')
plt.plot(data['Date'][train_size + time_step:train_size + time_step + len(predicted_high_test)], predicted_high_test, label='Predicted High', color='green', linestyle='--')
plt.plot(data['Date'][train_size + time_step:train_size + time_step + len(predicted_low_test)], predicted_low_test, label='Predicted Low', color='red', linestyle='--')
plt.plot(future_dates, predicted_highs, label='Future Predicted High', color='green', linestyle='--', marker='o')
plt.plot(future_dates, predicted_lows, label='Future Predicted Low', color='red', linestyle='--', marker='o')
plt.title('Actual and Future Predicted High/Low Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
