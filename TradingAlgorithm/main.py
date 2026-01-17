import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import os
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import root_mean_squared_error

# 1. SETUP
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = r'C:\Users\chrit\OneDrive\Documents\GitHub\stocktradingalgorithm\TradingAlgorithm\model_pro.pth'
os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

# 2. DATA PREPARATION (Adding Market Context)
ticker = input("Enter stock ticker (e.g., AAPL): ").upper()
# Get AAPL and S&P 500
data = yf.download([ticker, '^GSPC'], start='2000-01-01')['Close']

# Monthly Resample
df_monthly = data.resample('ME').last()

# Feature 1: AAPL Log Returns
df_features = pd.DataFrame(index=df_monthly.index)
df_features['AAPL_Returns'] = np.log(df_monthly[ticker] / df_monthly[ticker].shift(1))

# Feature 2: S&P 500 Log Returns (Market Context)
df_features['Market_Returns'] = np.log(df_monthly['^GSPC'] / df_monthly['^GSPC'].shift(1))

# Feature 3: Trend (Price vs 12-Month Moving Average)
ma12 = df_monthly[ticker].rolling(window=12).mean()
df_features['Trend'] = (df_monthly[ticker] - ma12) / ma12

# Feature 4 & 5: Volume and Volatility
df_daily = yf.download(ticker, start='2000-01-01')
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
    y.append(scaled_data[i + seq_length, 0]) # Target: AAPL Returns

X, y = np.array(X), np.array(y)
train_size = int(0.8 * len(X))

X_train = torch.FloatTensor(X[:train_size]).to(device)
y_train = torch.FloatTensor(y[:train_size]).unsqueeze(1).to(device)
X_test = torch.FloatTensor(X[train_size:]).to(device)
y_test = torch.FloatTensor(y[train_size:]).unsqueeze(1).to(device)

# 4. MODEL (Added more complexity)
class ProPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers):
        super(ProPredictor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, dropout=0.3)
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )
    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])

# input_dim is now 5
model = ProPredictor(5, 128, 2).to(device)
criterion = nn.HuberLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Learning Rate Scheduler: Lowers LR if the loss stops improving
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=20, factor=0.5)

# 5. TRAINING
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))

model.train()
for epoch in range(600):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    scheduler.step(loss) # Update scheduler
    if epoch % 100 == 0: print(f"Epoch {epoch}, Loss: {loss.item():.6f}, LR: {optimizer.param_groups[0]['lr']}")

torch.save(model.state_dict(), MODEL_PATH)

# 6. EVALUATION
model.eval()
with torch.no_grad():
    test_preds_scaled = model(X_test).cpu().numpy()
    dummy = np.zeros((len(test_preds_scaled), 5)) # Must be 5 features now
    dummy[:, 0] = test_preds_scaled.ravel()
    pred_log_returns = scaler.inverse_transform(dummy)[:, 0].ravel()

# One-Step Price Reconstruction
num_preds = len(pred_log_returns)
actual_prices_test = df_monthly[ticker].iloc[-num_preds:].values.ravel()
previous_actual_prices = df_monthly[ticker].iloc[-num_preds-1 : -1].values.ravel()
predicted_prices = previous_actual_prices * np.exp(pred_log_returns)

# 7. METRICS
rmse = root_mean_squared_error(actual_prices_test, predicted_prices)
print(f"\nPRO MODEL RMSE: ${rmse:.2f}")
print(f"Error Percentage: {(rmse / np.mean(actual_prices_test)) * 100:.2f}%")

# 8. PLOT
plt.figure(figsize=(12,6))
plt.plot(df_monthly.index[-num_preds:], actual_prices_test, label="Actual AAPL")
plt.plot(df_monthly.index[-num_preds:], predicted_prices, label="Pro Prediction", linestyle='--')
plt.title("Pro Model: AAPL Prediction with Market Context")
plt.legend()
plt.show()