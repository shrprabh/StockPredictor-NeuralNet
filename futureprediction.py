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

# Initialize scalers for each feature
scaler_close = MinMaxScaler(feature_range=(0, 1))
scaler_high = MinMaxScaler(feature_range=(0, 1))
scaler_low = MinMaxScaler(feature_range=(0, 1))

# Fit and transform the prices
scaled_close = scaler_close.fit_transform(data[['Close']])
scaled_high = scaler_high.fit_transform(data[['High']])
scaled_low = scaler_low.fit_transform(data[['Low']])

# Define the time step for previous days' data to use for prediction
time_step = 3

# Function to create the dataset for LSTM
def create_dataset(X, time_step):
    Xs, Ys = [], []
    for i in range(len(X) - time_step):
        v = X[i:(i + time_step), :]
        Xs.append(v)
        Ys.append(X[i + time_step, :])
    return np.array(Xs), np.array(Ys)

# Prepare the training data
X_train, y_train = create_dataset(np.hstack((scaled_close, scaled_high, scaled_low)), time_step)

# Reshape the input to be [samples, time steps, features] for the LSTM
X_train = X_train.reshape(X_train.shape[0], time_step, 3)

# Create the LSTM model
model = Sequential()
model.add(Bidirectional(LSTM(100, activation='tanh', return_sequences=True), input_shape=(time_step, 3)))
model.add(Dropout(0.3))
model.add(Bidirectional(LSTM(100, activation='tanh')))
model.add(Dropout(0.3))
model.add(Dense(3))  # Predicting three values: Close, High and Low
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
history = model.fit(X_train, y_train, epochs=100, batch_size=32)

# Prediction for the test dataset
train_size = int(len(scaled_close) * 0.8)
X_test, y_test = create_dataset(np.hstack((scaled_close[train_size:], scaled_high[train_size:], scaled_low[train_size:])), time_step)
X_test = X_test.reshape(X_test.shape[0], time_step, 3)
predicted_prices = model.predict(X_test)

# Inverse transform to get actual values
predicted_prices_transformed = np.hstack((scaler_close.inverse_transform(predicted_prices[:, 0].reshape(-1, 1)),
                                          scaler_high.inverse_transform(predicted_prices[:, 1].reshape(-1, 1)),
                                          scaler_low.inverse_transform(predicted_prices[:, 2].reshape(-1, 1))))
actual_prices = np.hstack((data['Close'][train_size + time_step:].values.reshape(-1, 1),
                           data['High'][train_size + time_step:].values.reshape(-1, 1),
                           data['Low'][train_size + time_step:].values.reshape(-1, 1)))

# Predict the next 10 days
# last_sequence = np.hstack((scaled_close, scaled_high, scaled_low))[-time_step:]
# future_dates = pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=10)
# future_predicted_prices = []

# for _ in range(10):
#     last_sequence_input = last_sequence.reshape((1, time_step, 3))
#     predicted_price = model.predict(last_sequence_input)
#     future_predicted_prices.append(predicted_price[0])
#     last_sequence = np.vstack((last_sequence[1:], predicted_price))

# Assuming the last data point is the most recent trading day
last_sequence = np.hstack((scaled_close, scaled_high, scaled_low))[-time_step:]

# Set up future dates for the next Monday to Saturday
future_dates = pd.date_range(data['Date'].iloc[-1] + pd.Timedelta(days=1), periods=6, freq='B')  # 'B' stands for business day frequency

future_predicted_prices = []

for _ in range(6):  # Predict for the next 6 business days
    last_sequence_input = last_sequence.reshape((1, time_step, 3))
    predicted_price = model.predict(last_sequence_input)
    future_predicted_prices.append(predicted_price[0])
    last_sequence = np.vstack((last_sequence[1:], predicted_price))

# Inverse transform the scaled predictions back to actual values
future_predicted_prices = np.array(future_predicted_prices)
future_predicted_close = scaler_close.inverse_transform(future_predicted_prices[:, 0].reshape(-1, 1))
future_predicted_high = scaler_high.inverse_transform(future_predicted_prices[:, 1].reshape(-1, 1))
future_predicted_low = scaler_low.inverse_transform(future_predicted_prices[:, 2].reshape(-1, 1))

# Plotting the actual and predicted prices
plt.figure(figsize=(15, 7))
plt.plot(data['Date'], data['Close'], label='Actual Close', color='blue')
plt.plot(data['Date'], data['High'], label='Actual High', color='orange')
plt.plot(data['Date'][train_size + time_step:], predicted_prices_transformed[:, 0], label='Predicted Close', color='green', linestyle='--')
plt.plot(data['Date'][train_size + time_step:], predicted_prices_transformed[:, 1], label='Predicted High', color='red', linestyle='--')
plt.plot(future_dates, future_predicted_close, label='Future Predicted Close', color='green', linestyle='--', marker='o')
plt.plot(future_dates, future_predicted_high, label='Future Predicted High', color='red', linestyle='--', marker='o')
plt.title('Actual and Future Predicted Close/High Prices')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.show()
