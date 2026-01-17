import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import yfinance as yf
from sklearn.metrics import mean_absolute_error, mean_squared_error
from collections import Counter

print("All libraries loaded")

config = {
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
        "symbol": "DJI",  # Dow Jones Industrial Average
        "period": "max",  # max, 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd
        "predict_days_ahead": 1,  # Predict 1 day ahead
    },
    "plots": {
        "xticks_interval": 90,
        "color_actual": "#001f3f",
        "color_train": "#3D9970",
        "color_val": "#0074D9",
        "color_pred_train": "#3D9970",
        "color_pred_val": "#0074D9",
        "color_pred_test": "#FF4136",
    },
    "model": {
        "input_size": 5,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.4,  # Even more dropout
    },
    "training": {
        "device": "cpu",
        "batch_size": 32,
        "num_epoch": 60,
        "learning_rate": 0.0005,  # Even lower LR
        "scheduler_step_size": 25,
        "weight_decay": 1e-4,  # Stronger L2
        "use_class_weights": True,  # NEW: Balance UP/DOWN
    }
}

def create_features(data):
    """Create simplified but robust features"""
    df = pd.DataFrame()
    
    df['returns'] = data['Close'].pct_change()
    df['volatility'] = df['returns'].rolling(10).std()
    df['momentum'] = data['Close'].pct_change(5)
    df['volume_change'] = data['Volume'].pct_change()
    
    ma20 = data['Close'].rolling(20).mean()
    df['price_to_ma20'] = (data['Close'] - ma20) / ma20
    
    df = df.fillna(0)
    
    # Clip outliers
    for col in df.columns:
        mean = df[col].mean()
        std = df[col].std()
        df[col] = df[col].clip(mean - 3*std, mean + 3*std)
    
    return df

def download_data(config):
    ticker = yf.Ticker(config["data"]["symbol"])
    data = ticker.history(period=config["data"]["period"])
    
    data_date = [date.strftime('%Y-%m-%d') for date in data.index]
    
    features_df = create_features(data)
    data_features = features_df.values
    config["model"]["input_size"] = data_features.shape[1]
    
    data_close_price = data['Close'].values
    
    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)
    print(f"Using {config['model']['input_size']} features")

    return data_date, data_close_price, data_features, num_data_points, display_date_range

data_date, data_close_price, data_features, num_data_points, display_date_range = download_data(config)

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
plt.plot(data_date, data_close_price, color=config["plots"]["color_actual"])
xticks = [data_date[i] if ((i%config["plots"]["xticks_interval"]==0 and (num_data_points-i) > config["plots"]["xticks_interval"]) or i==num_data_points-1) else None for i in range(num_data_points)]
x = np.arange(0,len(xticks))
plt.xticks(x, xticks, rotation='vertical')
plt.title("Daily close price for " + config["data"]["symbol"] + ", " + display_date_range)
plt.grid(which='major', axis='y', linestyle='--')
plt.show()

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        self.sd = np.where(self.sd == 0, 1, self.sd)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

scaler = Normalizer()
normalized_features = scaler.fit_transform(data_features)

# Validate normalized features
if np.any(np.isnan(normalized_features)) or np.any(np.isinf(normalized_features)):
    print("Warning: NaN or Inf values detected in normalized features. Replacing with 0...")
    normalized_features = np.nan_to_num(normalized_features, nan=0.0, posinf=0.0, neginf=0.0)

print(f"Normalized features shape: {normalized_features.shape}")
print(f"Normalized features stats: min={normalized_features.min():.4f}, max={normalized_features.max():.4f}, mean={normalized_features.mean():.4f}")

def prepare_data_x(x, window_size):
    # Ensure x is a 2D array
    if x.ndim == 1:
        x = x.reshape(-1, 1)
    
    # Check if we have enough data
    if x.shape[0] < window_size:
        raise ValueError(f"Not enough data points. Need at least {window_size}, got {x.shape[0]}")
    
    n_row = x.shape[0] - window_size + 1
    if n_row <= 0:
        raise ValueError(f"Window size {window_size} is too large for data of length {x.shape[0]}")
    
    # Ensure x is contiguous for as_strided
    if not x.flags['C_CONTIGUOUS']:
        x = np.ascontiguousarray(x)
    
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size, x.shape[1]), 
                                              strides=(x.strides[0], x.strides[0], x.strides[1]))
    return output[:-1], output[-1]

def prepare_data_y(prices, window_size, days_ahead=1):
    """Predict percentage return N days ahead"""
    current_prices = prices[window_size-1:-(days_ahead)]
    future_prices = prices[window_size+days_ahead-1:]
    returns = (future_prices - current_prices) / current_prices
    return returns

data_x, data_x_unseen = prepare_data_x(normalized_features, window_size=config["data"]["window_size"])
data_y = prepare_data_y(data_close_price, window_size=config["data"]["window_size"], 
                        days_ahead=config["data"]["predict_days_ahead"])

print(f"Data X shape: {data_x.shape}")
print(f"Data Y shape: {data_y.shape}")

# Validate shapes match
if data_x.shape[0] != data_y.shape[0]:
    raise ValueError(f"Shape mismatch: data_x has {data_x.shape[0]} samples but data_y has {data_y.shape[0]} samples")

# Ensure data_y is 1D
if data_y.ndim > 1:
    data_y = data_y.flatten()
    print(f"Flattened data_y to shape: {data_y.shape}")

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return (self.x[idx], self.y[idx])

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_up = np.sum(data_y_train > 0)
train_down = np.sum(data_y_train < 0)
print(f"Training set balance: UP={train_up} ({100*train_up/len(data_y_train):.1f}%), DOWN={train_down} ({100*train_down/len(data_y_train):.1f}%)")

val_up = np.sum(data_y_val > 0)
val_down = np.sum(data_y_val < 0)
print(f"Validation set balance: UP={val_up} ({100*val_up/len(data_y_val):.1f}%), DOWN={val_down} ({100*val_down/len(data_y_val):.1f}%)")

# Create weighted sampler to balance UP/DOWN during training
if config["training"]["use_class_weights"]:
    train_directions = np.sign(data_y_train)
    class_counts = Counter(train_directions)
    class_weights = {cls: 1.0/count for cls, count in class_counts.items()}
    sample_weights = [class_weights[cls] for cls in train_directions]
    # Convert to torch tensor and ensure proper type
    sample_weights = torch.DoubleTensor(sample_weights)
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], sampler=sampler, num_workers=0)
    print("Using weighted sampling to balance UP/DOWN training")
else:
    train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True, num_workers=0)

val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False, num_workers=0)

class LSTMModel(nn.Module):
    def __init__(self, input_size=1, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size

        self.lstm = nn.LSTM(input_size, hidden_size=self.hidden_layer_size, num_layers=num_layers,
                            batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.dropout = nn.Dropout(dropout)
        
        # Add batch normalization to help with training stability
        self.bn = nn.BatchNorm1d(hidden_layer_size)
        self.linear = nn.Linear(hidden_layer_size, output_size)

        self.init_weights()

    def init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)

    def forward(self, x):
        lstm_out, (h_n, c_n) = self.lstm(x)
        x = h_n[-1]
        
        # Only apply batch norm during training with batch size > 1
        # Batch norm requires batch size > 1, so skip if batch size is 1
        if self.training and x.shape[0] > 1:
            try:
                x = self.bn(x)
            except RuntimeError:
                # Skip batch norm if it fails (e.g., batch size 1)
                pass
        
        x = self.dropout(x)
        predictions = self.linear(x)
        return predictions.squeeze()

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0
    num_batches = len(dataloader)

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        try:
            if is_training:
                optimizer.zero_grad()

            batchsize = x.shape[0]

            x = x.to(config["training"]["device"])
            y = y.to(config["training"]["device"])

            out = model(x)
            loss = criterion(out, y)

            if is_training:
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            epoch_loss += (loss.detach().item() / batchsize)
        except Exception as e:
            print(f"\nError in batch {idx+1}/{num_batches}: {e}")
            raise

    lr = scheduler.get_last_lr()[0]

    return epoch_loss, lr

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.HuberLoss(delta=0.01)
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], 
                       weight_decay=config["training"]["weight_decay"])
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

print("\n" + "="*50)
print("TRAINING STARTED")
print("="*50)

best_val_loss = float('inf')
patience = 15
patience_counter = 0

for epoch in range(config["training"]["num_epoch"]):
    print(f"Epoch {epoch+1}/{config['training']['num_epoch']} - Training...", end=' ', flush=True)
    loss_train, lr_train = run_epoch(train_dataloader, is_training=True)
    print("Done. Validating...", end=' ', flush=True)
    loss_val, lr_val = run_epoch(val_dataloader)
    print("Done.")
    scheduler.step()
    
    if loss_val < best_val_loss:
        best_val_loss = loss_val
        patience_counter = 0
    else:
        patience_counter += 1
    
    if patience_counter >= patience:
        print(f"Early stopping at epoch {epoch+1}")
        break

    # Print every epoch for better visibility
    print('Epoch[{}/{}] | loss train:{:.6f}, val:{:.6f} | lr:{:.6f} | patience:{}/{}'
          .format(epoch + 1, config["training"]["num_epoch"], loss_train, loss_val, lr_train, patience_counter, patience))

print("="*50)
print("TRAINING COMPLETED")
print("="*50 + "\n")

# Evaluation
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

predicted_val_returns = np.array([])
actual_val_returns = np.array([])
for idx, (x, y) in enumerate(val_dataloader):
    x = x.to(config["training"]["device"])
    with torch.no_grad():
        out = model(x)
    out = out.cpu().detach().numpy()
    predicted_val_returns = np.concatenate((predicted_val_returns, out))
    actual_val_returns = np.concatenate((actual_val_returns, y.cpu().numpy()))

def returns_to_prices(returns, initial_price):
    prices = [initial_price]
    for ret in returns:
        prices.append(prices[-1] * (1 + ret))
    return np.array(prices[1:])

val_start_price = data_close_price[split_index + config["data"]["window_size"] - 1]
predicted_val_prices = returns_to_prices(predicted_val_returns, val_start_price)
actual_val_prices = data_close_price[split_index+config["data"]["window_size"]+config["data"]["predict_days_ahead"]-1:
                                     split_index+config["data"]["window_size"]+config["data"]["predict_days_ahead"]-1+len(predicted_val_returns)]

print("="*50)
print("PERFORMANCE METRICS")
print("="*50)

val_mae = mean_absolute_error(actual_val_prices, predicted_val_prices)
val_rmse = np.sqrt(mean_squared_error(actual_val_prices, predicted_val_prices))
val_mape = np.mean(np.abs((actual_val_prices - predicted_val_prices) / actual_val_prices)) * 100

print(f"\nValidation Set - Price Prediction:")
print(f"  MAE:  ${val_mae:.2f}")
print(f"  RMSE: ${val_rmse:.2f}")
print(f"  MAPE: {val_mape:.2f}%")

actual_directions = np.sign(actual_val_returns)
predicted_directions = np.sign(predicted_val_returns)
direction_accuracy = np.mean(actual_directions == predicted_directions) * 100

print(f"\n DIRECTION ACCURACY: {direction_accuracy:.2f}%")
print(f"   (Random = 50%, Good = 55-60%, Excellent = 60%+)")

correct_up = np.sum((actual_directions > 0) & (predicted_directions > 0))
correct_down = np.sum((actual_directions < 0) & (predicted_directions < 0))
total_up = np.sum(actual_directions > 0)
total_down = np.sum(actual_directions < 0)

predicted_up_count = np.sum(predicted_directions > 0)
predicted_down_count = np.sum(predicted_directions < 0)

print(f"\nDetailed Direction Analysis:")
print(f"  Correctly predicted UP:   {correct_up}/{total_up} ({100*correct_up/max(total_up,1):.1f}%)")
print(f"  Correctly predicted DOWN: {correct_down}/{total_down} ({100*correct_down/max(total_down,1):.1f}%)")

print(f"\nPrediction Bias Check:")
print(f"  Model predicted UP:   {predicted_up_count} times ({100*predicted_up_count/len(predicted_directions):.1f}%)")
print(f"  Model predicted DOWN: {predicted_down_count} times ({100*predicted_down_count/len(predicted_directions):.1f}%)")
print(f"  Actual UP:   {total_up} times ({100*total_up/len(actual_directions):.1f}%)")
print(f"  Actual DOWN: {total_down} times ({100*total_down/len(actual_directions):.1f}%)")

bias_diff = abs(predicted_up_count/len(predicted_directions) - total_up/len(actual_directions))
if bias_diff > 0.15:
    print(f"  WARNING: Model has {bias_diff*100:.1f}% prediction bias!")
else:
    print(f"  Model predictions are well-balanced!")

# Trading simulation
returns_if_traded = []
for i in range(len(predicted_val_returns)):
    if predicted_directions[i] > 0:
        returns_if_traded.append(actual_val_returns[i])
    elif predicted_directions[i] < 0:
        returns_if_traded.append(-actual_val_returns[i])
    else:
        returns_if_traded.append(0)

if len(returns_if_traded) > 0:
    avg_return = np.mean(returns_if_traded)
    std_return = np.std(returns_if_traded)
    sharpe = (avg_return / std_return) * np.sqrt(252) if std_return > 0 else 0
    total_return = np.prod([1 + r for r in returns_if_traded]) - 1
    
    print(f"\nSimulated Trading Performance:")
    print(f"  Average daily return: {avg_return*100:.3f}%")
    print(f"  Sharpe Ratio: {sharpe:.2f}")
    print(f"  Total return: {total_return*100:.2f}%")
    
    if sharpe > 1.0:
        print(f"   Good Sharpe - model shows promise!")
    elif sharpe > 0.5:
        print(f"   Moderate Sharpe - better than random")
    else:
        print(f"   Low Sharpe - marginal improvement")

print("="*50 + "\n")

fig = figure(figsize=(25, 5), dpi=80)
fig.patch.set_facecolor((1.0, 1.0, 1.0))
val_dates = data_date[split_index+config["data"]["window_size"]+config["data"]["predict_days_ahead"]-1:
                      split_index+config["data"]["window_size"]+config["data"]["predict_days_ahead"]-1+len(predicted_val_returns)]
plt.plot(val_dates, actual_val_prices, label="Actual prices", color=config["plots"]["color_actual"], linewidth=2)
plt.plot(val_dates, predicted_val_prices, label="Predicted prices", color=config["plots"]["color_pred_val"], linewidth=2, linestyle='--')
plt.title(f"{config['data']['symbol']} - Validation (Dir: {direction_accuracy:.1f}%, Sharpe: {sharpe:.2f})")
xticks = [val_dates[i] if ((i%20==0 and (len(val_dates)-i) > 20) or i==len(val_dates)-1) else None for i in range(len(val_dates))]
xs = np.arange(0,len(xticks))
plt.xticks(xs, xticks, rotation='vertical')
plt.xlabel("Date")
plt.ylabel("Price ($)")
plt.grid(which='major', axis='y', linestyle='--')
plt.legend()
plt.tight_layout()
plt.show()

# Predict next day
model.eval()

x = torch.tensor(data_x_unseen).float().to(config["training"]["device"]).unsqueeze(0)  # [batch, sequence, features]
with torch.no_grad():
    predicted_return = model(x)
predicted_return = predicted_return.cpu().detach().numpy()[0]

# Convert return to price
last_price = data_close_price[-1]
predicted_next_price = last_price * (1 + predicted_return)

print(f"Predicted close price of the next trading day: ${predicted_next_price:.2f}")
print(f"Predicted return: {predicted_return*100:.2f}%")

# Save predicted vs actual validation data to CSV
print("\nSaving predicted vs actual validation data to CSV...")
# Ensure data is properly formatted as arrays
actual_prices_array = np.array(actual_val_prices).flatten()
predicted_prices_array = np.array(predicted_val_prices).flatten()
dates_array = np.array(val_dates).flatten()

validation_df = pd.DataFrame({
    'Date': dates_array,
    'Actual_Price': actual_prices_array,
    'Predicted_Price': predicted_prices_array
})
validation_df.to_csv('predicted_vs_actual_validation.csv', index=False)
print("Saved to: predicted_vs_actual_validation.csv")
print(f"Saved {len(validation_df)} rows of predicted vs actual data")

# Predict next month (approximately 20-22 trading days)
print("\nPredicting next month stock movement...")
model.eval()

# Start with the last window of normalized features
current_window = data_x_unseen.copy()
predicted_returns = []
trading_days_in_month = 22  # Approximate trading days in a month

# Track prices for feature calculation
predicted_prices = [data_close_price[-1]]  # Start with last actual price

for day in range(trading_days_in_month):
    x = torch.tensor(current_window).float().to(config["training"]["device"]).unsqueeze(0)
    with torch.no_grad():
        predicted_return = model(x)
    predicted_return_value = predicted_return.cpu().detach().numpy()[0]
    predicted_returns.append(predicted_return_value)
    
    # Calculate next price from return
    next_price = predicted_prices[-1] * (1 + predicted_return_value)
    predicted_prices.append(next_price)
    
    # Create new feature row (simplified - using predicted return as proxy for features)
    # In a real scenario, you'd recalculate all features from the new price
    # For now, we'll use a simplified approach: shift window and add new normalized features
    # This is approximate but functional
    new_features = current_window[-1].copy()  # Start with last features
    # Update with new return-based features (simplified)
    new_features[0] = predicted_return_value  # returns
    # Other features would need to be recalculated, but for prediction we'll approximate
    
    # Update window: remove first element and add new features
    current_window = np.vstack([current_window[1:], new_features])

# Get the last actual price
last_actual_price = data_close_price[-1]
# Get the predicted price at the end of next month
predicted_price_end_month = predicted_prices[-1]

# Determine if stock will go up or down
price_change = predicted_price_end_month - last_actual_price
price_change_percent = (price_change / last_actual_price) * 100
direction = "up" if price_change > 0 else "down"

print(f"Last actual price: ${last_actual_price:.2f}")
print(f"Predicted price at end of next month: ${predicted_price_end_month:.2f}")
print(f"Predicted change: ${price_change:.2f} ({price_change_percent:.2f}%)")
print(f"Prediction: Stock will go {direction} in the next month")

# Save next month prediction to CSV
print("\nSaving next month prediction to CSV...")
prediction_df = pd.DataFrame({
    'Day': range(1, trading_days_in_month + 1),
    'Predicted_Price': predicted_prices[1:]  # Skip first element (starting price)
})
prediction_df.to_csv('next_month_prediction.csv', index=False)
print("Saved to: next_month_prediction.csv")

# Save simple direction prediction (just "up" or "down") as CSV
print("\nSaving final month direction prediction...")
direction_df = pd.DataFrame({
    'Direction': [direction]
})
direction_df.to_csv('stock_direction_prediction.csv', index=False)
print(f"Saved direction prediction to: stock_direction_prediction.csv")
print(f"Direction: {direction}")