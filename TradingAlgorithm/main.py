import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import root_mean_squared_error

# 1. SETUP DEVICE AND PATHS
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')


MODEL_PATH = r'C:\Users\chrit\OneDrive\Documents\GitHub\stocktradingalgorithm\TradingAlgorithm\model.pth'


os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 2. DATA PREPARATION
ticker = 'AAPL'
df = yf.download(ticker, start='2020-01-01')

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df[['Close']])

seq_length = 60
data_x = []
data_y = []

for i in range(len(df_scaled) - seq_length):
    data_x.append(df_scaled[i : i + seq_length - 1]) 
    data_y.append(df_scaled[i + seq_length - 1])     

data_x = np.array(data_x)
data_y = np.array(data_y)

train_size = int(0.8 * len(data_x))

X_train = torch.from_numpy(data_x[:train_size]).type(torch.Tensor).to(device)
y_train = torch.from_numpy(data_y[:train_size]).type(torch.Tensor).to(device)
X_test = torch.from_numpy(data_x[train_size:]).type(torch.Tensor).to(device)
y_test = torch.from_numpy(data_y[train_size:]).type(torch.Tensor).to(device)

# 3. MODEL DEFINITION
class PredictionModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(PredictionModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim, device=device)
        out, (hn, cn) = self.lstm(x, (h0.detach(), c0.detach()))
        out = self.fc(out[:, -1, :])
        return out

# 4. INITIALIZE MODEL, LOSS, AND OPTIMIZER
model = PredictionModel(input_dim=1, hidden_dim=32, num_layers=2, output_dim=1).to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)
num_epochs = 300 

# 5. LOAD -> TRAIN -> SAVE LOGIC
model_exists = os.path.exists(MODEL_PATH)

if model_exists:
    print("--- Loading existing model weights to continue training ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
else:
    print("--- No saved model found. Training new model from scratch ---")

print(f"Training/Updating model for {num_epochs} epochs...")
model.train() 
for i in range(num_epochs):
    y_train_pred = model(X_train)
    loss = criterion(y_train_pred, y_train)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    if i % 25 == 0:
        print(f'Epoch {i}, Loss: {loss.item()}')

# Save the updated weights
torch.save(model.state_dict(), MODEL_PATH)

# Print status message based on whether the file already existed
if model_exists:
    print("Updated existing model file.")
else:
    print("Created new model file.")

# 6. PREDICTION AND EVALUATION
model.eval() 
with torch.no_grad():
    y_train_pred = model(X_train)
    y_test_pred = model(X_test)

y_train_pred_np = scaler.inverse_transform(y_train_pred.cpu().numpy())
y_train_np = scaler.inverse_transform(y_train.cpu().numpy())
y_test_pred_np = scaler.inverse_transform(y_test_pred.cpu().numpy())
y_test_np = scaler.inverse_transform(y_test.cpu().numpy())

train_rmse = root_mean_squared_error(y_train_np, y_train_pred_np)
test_rmse = root_mean_squared_error(y_test_np, y_test_pred_np)

print(f"Train RMSE: {train_rmse:.4f}")
print(f"Test RMSE: {test_rmse:.4f}")

# 7. PLOTTING
fig = plt.figure(figsize=(12,10))
gs = fig.add_gridspec(4,1)

ax1 = fig.add_subplot(gs[:3,0])
test_dates = df.index[-len(y_test_np):]
ax1.plot(test_dates, y_test_np, color='blue', label='Actual Price')
ax1.plot(test_dates, y_test_pred_np, color='green', label='Predicted Price')
ax1.legend()
plt.title(f"{ticker} Stock Price Prediction (Updated Model)")

ax2 = fig.add_subplot(gs[3,0])
ax2.axhline(test_rmse, color='blue', linestyle='--', label='RMSE')
ax2.plot(test_dates, abs(y_test_np - y_test_pred_np), 'r', label='Prediction Error')
ax2.legend()
plt.tight_layout()
plt.show()