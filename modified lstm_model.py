import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping

# Load data
data = pd.read_csv('predictive_maintenance_dataset.csv')
data['date'] = pd.to_datetime(data['date'])

# Convert categorical columns to numeric
label_encoder = LabelEncoder()
data['device'] = label_encoder.fit_transform(data['device'])

# Select only numeric columns for prediction
numeric_columns = ['device', 'failure', 'metric1', 'metric2', 'metric3', 
                  'metric4', 'metric5', 'metric6', 'metric7', 'metric8', 'metric9']
data_numeric = data[numeric_columns]

# Normalize data
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(data_numeric)

# Prepare data for LSTM
X, y = [], []
for i in range(60, len(scaled_data)):
    X.append(scaled_data[i-60:i])
    y.append(scaled_data[i])

X, y = np.array(X), np.array(y)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build LSTM model
model = Sequential()
model.add(LSTM(50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(50))
model.add(Dense(scaled_data.shape[1]))  # Changed to match the number of features

early_stop = EarlyStopping(monitor='loss', patience=3)
model.compile(optimizer='adam', loss='mean_squared_error')
model.fit(X_train, y_train, epochs=100, batch_size=32, callbacks=[early_stop])

# Create models directory if it doesn't exist
import os
os.makedirs('models', exist_ok=True)

# Save model
model.save('models/lstm_model.h5')
