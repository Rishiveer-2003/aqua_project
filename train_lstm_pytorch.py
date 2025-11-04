# train_lstm_pytorch.py - LSTM Time-Series Model Training Script (PyTorch version)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import joblib
import os

print(f"Using PyTorch version: {torch.__version__}")

# --- 1. Load Data ---
DATA_FILE = 'dynamic_flood_data.csv'
if not os.path.exists(DATA_FILE):
    print(f"Error: '{DATA_FILE}' not found.")
    print("Please run 'create_dynamic_dataset.py' first to generate it.")
    exit()

df = pd.read_csv(DATA_FILE)
print(f"Successfully loaded {len(df)} rows from {DATA_FILE}.")

# --- 2. Define Features (X) and Target (y) ---
feature_cols = [f'rain_lag_{i}' for i in range(1, 8)]
target_col = 'FloodProbability_Proxy'

if not all(col in df.columns for col in feature_cols + [target_col]):
    print("Error: The CSV file is missing required columns.")
    print(f"Ensure it has: {feature_cols} and {target_col}")
    exit()

X = df[feature_cols].values
y = df[target_col].values

# --- 3. Scale the Data ---
scaler_X = MinMaxScaler()
scaler_y = MinMaxScaler()

X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).flatten()

joblib.dump(scaler_X, 'lstm_scaler_X.pkl')
joblib.dump(scaler_y, 'lstm_scaler_y.pkl')
print("Data scalers created and saved.")

# --- 4. Time-Sequential Train/Test Split ---
split_ratio = 0.8
split_index = int(len(X_scaled) * split_ratio)

X_train = X_scaled[:split_index]
y_train = y_scaled[:split_index]
X_test = X_scaled[split_index:]
y_test = y_scaled[split_index:]

print(f"Data split: {len(X_train)} training, {len(X_test)} testing samples.")

# --- 5. Reshape for LSTM (samples, timesteps, features) ---
X_train_lstm = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
X_test_lstm = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

# Convert to PyTorch tensors
X_train_tensor = torch.FloatTensor(X_train_lstm)
y_train_tensor = torch.FloatTensor(y_train).view(-1, 1)
X_test_tensor = torch.FloatTensor(X_test_lstm)
y_test_tensor = torch.FloatTensor(y_test).view(-1, 1)

print(f"Reshaped X_train for LSTM: {X_train_tensor.shape}")
print(f"Reshaped X_test for LSTM: {X_test_tensor.shape}")

# --- 6. Define LSTM Model ---
class FloodLSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=50, num_layers=2, output_size=1):
        super(FloodLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.2)
        self.fc1 = nn.Linear(hidden_size, 25)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(25, output_size)
        
    def forward(self, x):
        # LSTM layer
        lstm_out, _ = self.lstm(x)
        # Take the output from the last timestep
        out = lstm_out[:, -1, :]
        # Fully connected layers
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

# --- 7. Create Dataset class ---
class FloodDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

train_dataset = FloodDataset(X_train_tensor, y_train_tensor)
test_dataset = FloodDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=False)  # Don't shuffle time series
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# --- 8. Initialize Model ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

model = FloodLSTM(input_size=1, hidden_size=50, num_layers=2, output_size=1).to(device)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("\nModel Architecture:")
print(model)

# --- 9. Training Loop ---
num_epochs = 100
best_val_loss = float('inf')
patience = 10
patience_counter = 0

print("\nStarting model training...")

for epoch in range(num_epochs):
    # Training
    model.train()
    train_loss = 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        # Forward pass
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    train_loss /= len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            val_loss += loss.item()
    
    val_loss /= len(test_loader)
    
    print(f"Epoch [{epoch+1}/{num_epochs}] - Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
    
    # Early stopping check
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        # Save best model
        torch.save(model.state_dict(), 'lstm_model_best.pth')
        print(f"  ✓ Model saved (val_loss: {val_loss:.6f})")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"\nEarly stopping triggered after {epoch+1} epochs")
            break

print("\nTraining complete!")
print(f"Best model saved to 'lstm_model_best.pth'")

# --- 10. Load Best Model and Evaluate ---
model.load_state_dict(torch.load('lstm_model_best.pth', weights_only=True))
model.eval()

with torch.no_grad():
    train_pred = model(X_train_tensor.to(device)).cpu().numpy()
    test_pred = model(X_test_tensor.to(device)).cpu().numpy()

# Inverse transform predictions
train_pred = scaler_y.inverse_transform(train_pred)
test_pred = scaler_y.inverse_transform(test_pred)
y_train_actual = scaler_y.inverse_transform(y_train.reshape(-1, 1))
y_test_actual = scaler_y.inverse_transform(y_test.reshape(-1, 1))

# Calculate RMSE
from sklearn.metrics import mean_squared_error
train_rmse = np.sqrt(mean_squared_error(y_train_actual, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test_actual, test_pred))

print(f"\nFinal Metrics:")
print(f"Training RMSE: {train_rmse:.4f}")
print(f"Testing RMSE: {test_rmse:.4f}")
