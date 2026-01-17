import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import yfinance as yf

print("All libraries loaded")

# Get ticker from command line argument or use default
ticker = sys.argv[1] if len(sys.argv) > 1 else "IBM"

config = {
    "data": {
        "window_size": 20,
        "train_split_size": 0.80,
        "symbol": ticker,
        "period": "max",
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
        "input_size": 1,
        "num_lstm_layers": 2,
        "lstm_size": 32,
        "dropout": 0.2,
    },
    "training": {
        "device": "cpu",
        "batch_size": 64,
        "num_epoch": 30,
        "learning_rate": 0.01,
        "scheduler_step_size": 40,
    }
}

def download_data(config):
    print(f"Downloading data for {config['data']['symbol']}...")
    ticker_obj = yf.Ticker(config["data"]["symbol"])
    data = ticker_obj.history(period=config["data"]["period"])
    
    data_date = [date.strftime('%Y-%m-%d') for date in data.index]
    data_close_price = data['Close'].values
    
    num_data_points = len(data_date)
    display_date_range = "from " + data_date[0] + " to " + data_date[num_data_points-1]
    print("Number data points", num_data_points, display_date_range)

    return data_date, data_close_price, num_data_points, display_date_range

data_date, data_close_price, num_data_points, display_date_range = download_data(config)

class Normalizer():
    def __init__(self):
        self.mu = None
        self.sd = None

    def fit_transform(self, x):
        self.mu = np.mean(x, axis=(0), keepdims=True)
        self.sd = np.std(x, axis=(0), keepdims=True)
        normalized_x = (x - self.mu)/self.sd
        return normalized_x

    def inverse_transform(self, x):
        return (x*self.sd) + self.mu

scaler = Normalizer()
normalized_data_close_price = scaler.fit_transform(data_close_price)

def prepare_data_x(x, window_size):
    n_row = x.shape[0] - window_size + 1
    output = np.lib.stride_tricks.as_strided(x, shape=(n_row, window_size), strides=(x.strides[0], x.strides[0]))
    return output[:-1], output[-1]

def prepare_data_y(x, window_size):
    y = x[window_size:]
    return y

data_x, data_x_unseen = prepare_data_x(normalized_data_close_price, window_size=config["data"]["window_size"])
data_y = prepare_data_y(normalized_data_close_price, window_size=config["data"]["window_size"])

split_index = int(data_y.shape[0]*config["data"]["train_split_size"])
data_x_train = data_x[:split_index]
data_x_val = data_x[split_index:]
data_y_train = data_y[:split_index]
data_y_val = data_y[split_index:]

to_plot_data_y_train = np.zeros(num_data_points)
to_plot_data_y_val = np.zeros(num_data_points)

to_plot_data_y_train[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(data_y_train)
to_plot_data_y_val[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(data_y_val)

to_plot_data_y_train = np.where(to_plot_data_y_train == 0, None, to_plot_data_y_train)
to_plot_data_y_val = np.where(to_plot_data_y_val == 0, None, to_plot_data_y_val)

class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

dataset_train = TimeSeriesDataset(data_x_train, data_y_train)
dataset_val = TimeSeriesDataset(data_x_val, data_y_val)

print("Train data shape", dataset_train.x.shape, dataset_train.y.shape)
print("Validation data shape", dataset_val.x.shape, dataset_val.y.shape)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size, dropout):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(input_size, hidden_layer_size, num_layers, batch_first=True, dropout=dropout if num_layers > 1 else 0)

        self.linear = nn.Linear(hidden_layer_size, output_size)

    def forward(self, x):
        lstm_out, (cn, hn) = self.lstm(x)
        predictions = self.linear(lstm_out[:, -1])
        return predictions

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0

    if is_training:
        model.train()
    else:
        model.eval()

    for idx, (x, y) in enumerate(dataloader):
        if is_training:
            model.zero_grad()

        batchX = x.to(config["training"]["device"]).float()
        batchY = y.to(config["training"]["device"]).float()

        batchX = batchX.reshape((len(batchX), config["data"]["window_size"], -1))

        output = model(batchX)

        loss = criterion(output.squeeze(), batchY)

        if is_training:
            loss.backward()
            optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(dataloader)

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=True)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=True)

model = LSTMModel(input_size=config["model"]["input_size"], hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], output_size=1, dropout=config["model"]["dropout"])
model = model.to(config["training"]["device"])

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

print("Training model...")
for epoch in range(config["training"]["num_epoch"]):
    loss_train = run_epoch(train_dataloader, is_training=True)
    loss_val = run_epoch(val_dataloader, is_training=False)

    scheduler.step()

    if epoch % 10 == 0:
        print('Epoch[{}/{}] | loss train:{:.6f}, test:{:.6f}'.format(epoch, config["training"]["num_epoch"], loss_train, loss_val))

print("Model training complete")

train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

model.eval()

predicted_train = np.array([])

for idx, (x, y) in enumerate(train_dataloader):
    batchX = x.to(config["training"]["device"]).float()
    batchX = batchX.reshape((len(batchX), config["data"]["window_size"], -1))
    output = model(batchX)
    predicted_train = np.append(predicted_train, output.cpu().detach().numpy())

predicted_val = np.array([])

for idx, (x, y) in enumerate(val_dataloader):
    batchX = x.to(config["training"]["device"]).float()
    batchX = batchX.reshape((len(batchX), config["data"]["window_size"], -1))
    output = model(batchX)
    predicted_val = np.append(predicted_val, output.cpu().detach().numpy())

to_plot_data_y_train_pred = np.zeros(num_data_points)
to_plot_data_y_val_pred = np.zeros(num_data_points)

to_plot_data_y_train_pred[config["data"]["window_size"]:split_index+config["data"]["window_size"]] = scaler.inverse_transform(predicted_train)
to_plot_data_y_val_pred[split_index+config["data"]["window_size"]:] = scaler.inverse_transform(predicted_val)

to_plot_data_y_train_pred = np.where(to_plot_data_y_train_pred == 0, None, to_plot_data_y_train_pred)
to_plot_data_y_val_pred = np.where(to_plot_data_y_val_pred == 0, None, to_plot_data_y_val_pred)

# Save predicted vs actual validation data to CSV
print("\nSaving predicted vs actual validation data to CSV...")
validation_data = []
val_start_index = split_index + config["data"]["window_size"]

for i in range(len(predicted_val)):
    validation_data.append({
        'Date': data_date[val_start_index + i],
        'Actual_Price': float(scaler.inverse_transform(np.array([[data_y_val[i]]]))[0][0]),
        'Predicted_Price': float(to_plot_data_y_val_pred[val_start_index + i])
    })

validation_df = pd.DataFrame(validation_data)
validation_df.to_csv('predicted_vs_actual_validation.csv', index=False)
print("Saved to: predicted_vs_actual_validation.csv")

# Predict direction for next month
print("\nGenerating next month prediction...")
model.eval()

current_window = data_x_unseen.copy()
predicted_prices = []

trading_days_in_month = 22

for day in range(trading_days_in_month):
    x = torch.tensor(current_window).float().to(config["training"]["device"]).unsqueeze(0).unsqueeze(2)
    with torch.no_grad():
        predicted_normalized = model(x)
    
    predicted_normalized_np = predicted_normalized.cpu().detach().numpy()
    if predicted_normalized_np.ndim == 0:
        predicted_normalized_value = float(predicted_normalized_np)
    else:
        predicted_normalized_value = float(predicted_normalized_np[0])
    
    predicted_price = scaler.inverse_transform(np.array([[predicted_normalized_value]]))[0][0]
    predicted_prices.append(predicted_price)
    
    current_window = np.append(current_window[1:], predicted_normalized_value)

last_actual_price = data_close_price[-1]
predicted_price_end_month = predicted_prices[-1]

price_change = predicted_price_end_month - last_actual_price
price_change_percent = (price_change / last_actual_price) * 100
direction = "up" if price_change > 0 else "down"

print(f"Last actual price: ${last_actual_price:.2f}")
print(f"Predicted price at end of next month: ${predicted_price_end_month:.2f}")
print(f"Predicted change: ${price_change:.2f} ({price_change_percent:.2f}%)")
print(f"Prediction: Stock will go {direction} in the next month")

prediction_df = pd.DataFrame({
    'Day': range(1, trading_days_in_month + 1),
    'Predicted_Price': predicted_prices[1:]
})
prediction_df.to_csv('next_month_prediction.csv', index=False)
print("Saved to: next_month_prediction.csv")

direction_df = pd.DataFrame({
    'Direction': [direction]
})
direction_df.to_csv('stock_direction_prediction.csv', index=False)
print(f"Saved direction prediction to: stock_direction_prediction.csv")
print(f"Direction: {direction}")
