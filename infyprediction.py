import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout

# Load the dataset
data = pd.read_csv('INFY.NS-3.csv')

# Convert the date column to datetime and sort the data
data['Date'] = pd.to_datetime(data['Date'])
data.sort_values('Date', inplace=True)

# Initialize scalers for each feature
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_high = MinMaxScaler(feature_range=(0, 1))
scaler_low = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the 'Close', 'High', and 'Low' prices
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
model.add(LSTM(50, activation='relu', input_shape=(X_train.shape[1], X_train.shape[2]), return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(50, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(2))  # Predicting two values: High and Low
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Predicting the High and Low prices for the test data
train_size = int(len(scaled_close) * 0.8)
X_test, y_test = create_dataset(scaled_close[train_size:], scaled_high[train_size:], scaled_low[train_size:], time_step)
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
predicted_high_low_test = model.predict(X_test)
predicted_high_test = scaler_high.inverse_transform(predicted_high_low_test[:, 0].reshape(-1, 1))
predicted_low_test = scaler_low.inverse_transform(predicted_high_low_test[:, 1].reshape(-1, 1))

# Plot the model loss
plt.figure(figsize=(10, 4))
plt.plot(history.history['loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train'], loc='upper left')
# plt.show()

# Plot the actual and predicted high and low prices
plt.figure(figsize=(15, 7))
plt.plot(data['Date'], data['High'], label='Actual High', color='blue')
plt.plot(data['Date'], data['Low'], label='Actual Low', color='orange')
test_dates = data['Date'][train_size + time_step:]
plt.plot(test_dates, predicted_high_test, label='Predicted High', color='green', linestyle='--')
plt.plot(test_dates, predicted_low_test, label='Predicted Low', color='red', linestyle='--')
plt.title('Actual and Predicted High/Low Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
# plt.show()

# ... [previous code] ...

# Define the number of days you want to predict into the future
num_days_to_predict = 5

# Start with the last `time_step` days from the training set as the initial input
last_sequence = scaled_close[-time_step:]
new_predicted_high = []
new_predicted_low = []

# Predict the next `num_days_to_predict` days
for _ in range(num_days_to_predict):
    last_sequence = last_sequence.reshape((1, time_step, 1))
    next_day_high_low = model.predict(last_sequence)
    new_predicted_high.append(next_day_high_low[0, 0])
    new_predicted_low.append(next_day_high_low[0, 1])
    
    # Update the last sequence for the next prediction
    next_day_high_low = next_day_high_low.reshape((1, 1, 2)) # Reshape to be 3D
    last_sequence = np.append(last_sequence[:, 1:, :], next_day_high_low[:, :, :1], axis=1)

# Inverse transform the predicted values to the original scale
new_predicted_high = scaler_high.inverse_transform(np.array(new_predicted_high).reshape(-1, 1))
new_predicted_low = scaler_low.inverse_transform(np.array(new_predicted_low).reshape(-1, 1))

# Generate future dates for plotting
last_date = data['Date'].max()
future_dates = [last_date + pd.Timedelta(days=i) for i in range(1, num_days_to_predict + 1)]

# Plot the actual and future predicted high and low prices
plt.figure(figsize=(15, 7))
plt.plot(data['Date'], data['High'], label='Actual High', color='blue')
plt.plot(data['Date'], data['Low'], label='Actual Low', color='orange')
plt.plot(future_dates, new_predicted_high, label='Future Predicted High', color='green', linestyle='--', marker='o')
plt.plot(future_dates, new_predicted_low, label='Future Predicted Low', color='red', linestyle='--', marker='o')
plt.title('Actual and Future Predicted High/Low Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()

