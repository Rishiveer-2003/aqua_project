# train_lstm.py - LSTM Time-Series Model Training Script

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
import joblib
import os

print(f"Using TensorFlow version: {tf.__version__}")

# --- 1. Load Data ---
DATA_FILE = 'dynamic_flood_data.csv'
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found.")
    print("Please run 'create_dynamic_dataset.py' first to generate it.")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"Successfully loaded {len(df)} rows from {DATA_FILE}.")

# --- 2. Define Features (X) and Target (y) ---
# Our features are the 7 days of rainfall lag
feature_cols = [f'rain_lag_{i}' for i in range(1, 8)]
target_col = 'FloodProbability_Proxy'

# Ensure all columns exist
if not all(col in df.columns for col in feature_cols + [target_col]):
    print("Error: The CSV file is missing required columns.")
    print(f"Ensure it has: {feature_cols} and {target_col}")
    exit()

X = df[feature_cols].values
y = df[target_col].values

# --- 3. Scale the Data ---
# LSTMs perform best when data is scaled (usually to [0, 1])
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

# Fit and transform the scalers
X_scaled = scaler_X.fit_transform(X)
# We reshape 'y' to 2D for the scaler, then back to 1D
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

# Save these scalers! We need them for predictions later.
joblib.dump(scaler_X, 'lstm_scaler_X.pkl')
joblib.dump(scaler_y, 'lstm_scaler_y.pkl')
print("Data scalers created and saved as 'lstm_scaler_X.pkl' and 'lstm_scaler_y.pkl'.")

# --- 4. Create Time-Sequential Train/Test Split ---
# CRITICAL: We cannot shuffle time-series data.
# We must train on the "past" and test on the "future".
split_ratio = 0.8
split_index = int(len(X_scaled) * split_ratio)

X_train = X_scaled[:split_index]
y_train = y_scaled[:split_index]

X_test = X_scaled[split_index:]
y_test = y_scaled[split_index:]

print(f"Data split sequentially: {len(X_train)} training samples, {len(X_test)} testing samples.")

# --- 5. Reshape Data for LSTM ---
# The LSTM layer expects a 3D tensor in the shape:
# [samples, timesteps, features]
#
# Our current shape: (samples, 7 features)
# Our target shape: (samples, 7 timesteps, 1 feature per timestep)
#
# We are treating each of the 7 lag days as one "timestep"
# and the rainfall at that step as the "feature".

X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

print(f"Reshaped X_train for LSTM: {X_train_lstm.shape}") # e.g., (9216, 7, 1)
print(f"Reshaped X_test for LSTM: {X_test_lstm.shape}")   # e.g., (2304, 7, 1)

# --- 6. Build the LSTM Model ---
model = Sequential()

# Input layer: shape is (timesteps, features)
model.add(Input(shape=(X_train_lstm.shape[1], X_train_lstm.shape[2])))

# LSTM layer with 50 neurons. 
# 'return_sequences=True' is needed if you stack another LSTM layer
model.add(LSTM(units=50, return_sequences=True))
model.add(Dropout(0.2)) # Prevents overfitting

# A second LSTM layer for deeper pattern recognition
model.add(LSTM(units=50, return_sequences=False)) # False on the last LSTM layer
model.add(Dropout(0.2))

# A standard fully-connected layer
model.add(Dense(units=25, activation='relu'))

# Output layer: 1 neuron (our flood probability)
# We use 'linear' activation because this is a regression problem
model.add(Dense(units=1, activation='linear'))

# --- 7. Compile the Model ---
# We use 'mean_squared_error' as the loss, which aligns with your
# RMSE evaluation metric from your project report.
model.compile(optimizer='adam', loss='mean_squared_error')

model.summary()

# --- 8. Define Callbacks ---
# These help us save the best model and stop training when it's done
MODEL_FILE = 'lstm_model_best.keras'

# Save only the best version of the model
checkpoint = ModelCheckpoint(
    MODEL_FILE, 
    monitor='val_loss',     # Monitor the validation loss
    save_best_only=True,  # Only save if it's the best so far
    mode='min',             # We want to minimize the loss
    verbose=1
)

# Stop training if the model's validation loss stops improving
early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=10,          # Wait 10 epochs for improvement
    restore_best_weights=True # Restore the weights from the best epoch
)

# --- 9. Train the Model ---
print("\nStarting model training...")
history = model.fit(
    X_train_lstm, y_train,
    epochs=100,           # Max 100 epochs (EarlyStopping will likely stop it sooner)
    batch_size=32,        # Process 32 samples at a time
    validation_data=(X_test_lstm, y_test),
    callbacks=[checkpoint, early_stopping],
    verbose=1             # Show progress
)

print("\nTraining complete.")
print(f"Best model saved to '{MODEL_FILE}'.")
