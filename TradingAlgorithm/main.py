import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error # Importing the metric

# 1.
try:
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print("Using GPU: " + torch.cuda.get_device_name(0))
    else:
        device = torch.device('cpu')
        print("Using CPU (CUDA not available)")
except Exception:
    device = torch.device('cpu')
    print("Using CPU (Error in CUDA initialization)")
MODEL_PATH = r'C:\Users\chrit\OneDrive\Documents\GitHub\stocktradingalgorithm\TradingAlgorithm\model_returns.pth'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 2. DATA PREPARATION
ticker = 'AAPL'
df_daily = yf.download(ticker, start='2000-01-01')

# Calculate Monthly Log Returns
df_monthly = df_daily['Close'].resample('ME').last()
df_returns = np.log(df_monthly / df_monthly.shift(1)).dropna()

# Supporting features
df_features = pd.DataFrame(index=df_returns.index)
df_features['Returns'] = df_returns
df_features['Vol_Change'] = df_daily['Volume'].resample('ME').sum().pct_change()
df_features['Range'] = ((df_daily['High'] - df_daily['Low']) / df_daily['Close']).resample('ME').mean()
df_features.dropna(inplace=True)

# 3. SCALING
scaler = MinMaxScaler(feature_range=(-1, 1))
scaled_data = scaler.fit_transform(df_features)

seq_length = 12 
X, y = [], []

for i in range(len(scaled_data) - seq_length):
    X.append(scaled_data[i : i + seq_length])
    y.append(scaled_data[i + seq_length, 0])

X = np.array(X)
y = np.array(y)

train_size = int(0.8 * len(X))
X_train = torch.FloatTensor(X[:train_size]).to(device)
y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X[train_size:]).to(device)
y_test = torch.FloatTensor(y[train_size:]).unsqueeze(1).to(device)

# 4. MODEL
class ReturnPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ReturnPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.2)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.LeakyReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

model = ReturnPredictor(3, 64, 2).to(device)
criterion = nn.HuberLoss() 
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 5. TRAINING
if os.path.exists(MODEL_PATH):
    print("--- Loading existing model weights ---")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.train()
for epoch in range(500):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.6f}")

torch.save(model.state_dict(), MODEL_PATH)

# 6. EVALUATION & RMSE CALCULATION
model.eval()
with torch.no_grad():
    test_preds_scaled = model(X_test).cpu().numpy()
    dummy = np.zeros((len(test_preds_scaled), 3))
    dummy[:, 0] = test_preds_scaled.ravel()
    pred_log_returns = scaler.inverse_transform(dummy)[:, 0].ravel()
num_preds = len(pred_log_returns)
actual_prices_test = df_monthly.iloc[-num_preds:].values.ravel()
previous_actual_prices = df_monthly.iloc[-num_preds-1 : -1].values.ravel()
min_len = min(len(actual_prices_test), len(previous_actual_prices), len(pred_log_returns))
actual_prices_test = actual_prices_test[:min_len]
previous_actual_prices = previous_actual_prices[:min_len]
pred_log_returns = pred_log_returns[:min_len]
predicted_prices = previous_actual_prices * np.exp(pred_log_returns)

# 7. FINAL METRICS
rmse = root_mean_squared_error(actual_prices_test, predicted_prices)

print("-" * 30)
print(f"ROOT MEAN SQUARE ERROR (RMSE): ${rmse:.2f}")
print(f"Mean Stock Price in Test Set: ${np.mean(actual_prices_test):.2f}")
print(f"Error Percentage: {(rmse / np.mean(actual_prices_test)) * 100:.2f}%")
print("-" * 30)
# 8. PLOTTING
plt.figure(figsize=(12,7))

# Main Price Chart
plt.subplot(2, 1, 1)
test_dates = df_monthly.index[train_size + seq_length + 1:]
plt.plot(test_dates, actual_prices_test, label="Actual Price", color='blue', linewidth=2)
plt.plot(test_dates, predicted_prices, label="One-Step Prediction", color='red', linestyle='--', alpha=0.8)
plt.title(f"{ticker} Optimized Price Prediction (Monthly)")
plt.ylabel("Price ($)")
plt.legend()

# Error Chart
plt.subplot(2, 1, 2)
errors = actual_prices_test - predicted_prices
plt.fill_between(test_dates, errors, color='red', alpha=0.3, label='Price Error')
plt.axhline(0, color='black', lw=1)
plt.ylabel("Error ($)")
plt.xlabel("Date")
plt.legend()

plt.tight_layout()
plt.show()