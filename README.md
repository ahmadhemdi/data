# data
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt

# Load data from CSV file
data = pd.read_csv('/content/sample_data/INDF.JK.csv')

# Filter data based on date range
start_date = '2020-08-18'
end_date = '2023-08-18'
filtered_data = data[(data['Date'] >= start_date) & (data['Date'] <= end_date)]

# Use 'Close' prices as the feature for prediction
close_prices = filtered_data['Close'].values

# Normalize the data
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(close_prices.reshape(-1, 1))

# Prepare training data
x_train = []
y_train = []

# Create sequences for training data
for i in range(60, len(scaled_data)):
    x_train.append(scaled_data[i-60:i, 0])
    y_train.append(scaled_data[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

# Build the ANN model
model = Sequential()
model.add(Dense(units=8, activation='relu', input_dim=60))
model.add(Dense(units=1))
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(x_train, y_train, epochs=100, batch_size=32)

# Prepare test data
inputs = scaled_data[len(scaled_data) - len(y_train) - 60:]
x_test = []

for i in range(60, len(inputs)):
    x_test.append(inputs[i-60:i, 0])

x_test = np.array(x_test)
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1]))

# Predict stock prices using the model
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)

# Display the table of dates, actual prices, and predicted prices
result_table = pd.DataFrame({'Date': filtered_data['Date'][60:], 'Actual Price': close_prices[60:], 'Predicted Price': predicted_prices.flatten()})
print(result_table)

# Plot the actual and predicted prices
plt.figure(figsize=(12, 6))
plt.plot(filtered_data['Date'][60:], close_prices[60:], label='Actual Prices', color='blue')
plt.plot(filtered_data['Date'][60:], predicted_prices.flatten(), label='Predicted Prices', color='red')
plt.xlabel('Date')
plt.ylabel('Price')
plt.title('Actual vs. Predicted Stock Prices')
plt.legend()
plt.xticks(filtered_data['Date'][60::30], rotation=45)  # Set interval here (30 days in this case)
plt.tight_layout()

# Display actual and predicted prices at the end of the graph
actual_price_end = close_prices[-1]
predicted_price_end = predicted_prices[-1][0]
plt.text(filtered_data['Date'].iloc[-1], actual_price_end, f'Actual: {actual_price_end:.2f}', color='blue')
plt.text(filtered_data['Date'].iloc[-1], predicted_price_end, f'Predicted: {predicted_price_end:.2f}', color='red')

plt.show()
