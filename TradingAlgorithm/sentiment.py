import sys
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import yfinance as yf

# Import transformers
try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    print("⚠ Transformers not found.")
    TRANSFORMERS_AVAILABLE = False

from datetime import datetime, timedelta
import requests
import warnings
warnings.filterwarnings('ignore')

print("Core libraries loaded")

# Get ticker from command line argument or use default
ticker = sys.argv[1] if len(sys.argv) > 1 else "AAPL"

# Get target date from command line (format: YYYY-MM-DD) or use current date
target_date_str = sys.argv[2] if len(sys.argv) > 2 else None
if target_date_str:
    try:
        target_date = datetime.strptime(target_date_str, '%Y-%m-%d')
        print(f"📅 Using target date: {target_date.strftime('%Y-%m-%d')}")
    except ValueError:
        print(f"⚠ Invalid date format '{target_date_str}'. Use YYYY-MM-DD. Using current date.")
        target_date = datetime.now()
else:
    target_date = datetime.now()
    print(f"📅 Using current date: {target_date.strftime('%Y-%m-%d')}")

config = {
    "data": {"window_size": 20, "train_split_size": 0.80, "symbol": ticker, "period": "max", "target_date": target_date},
    "plots": {"xticks_interval": 90, "color_actual": "#001f3f", "color_train": "#3D9970", 
              "color_val": "#0074D9", "color_pred_train": "#3D9970", "color_pred_val": "#0074D9", 
              "color_pred_test": "#FF4136"},
    "model": {"input_size": 2, "num_lstm_layers": 2, "lstm_size": 32, "dropout": 0.2},
    "training": {"device": "cpu", "batch_size": 64, "num_epoch": 30, "learning_rate": 0.01, 
                 "scheduler_step_size": 40},
    "sentiment": {
        "model_name": "ProsusAI/finbert",
        "news_api_key": "f18c946706924f6f8f864648b5338afd",
        "use_real_api": True
    }
}

# Load FinBERT
if TRANSFORMERS_AVAILABLE:
    print("Loading FinBERT model (this may take a minute on first run)...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(config["sentiment"]["model_name"])
        finbert_model = AutoModelForSequenceClassification.from_pretrained(config["sentiment"]["model_name"])
        finbert_model.eval()
        print("✓ FinBERT loaded successfully")
    except Exception as e:
        print(f"⚠ Error loading FinBERT: {e}")
        print("  The model will download on first use (~500MB)")
        TRANSFORMERS_AVAILABLE = False
else:
    print("⚠ Transformers not available. Sentiment analysis disabled.")
    print("  Install with: pip install transformers torch")
    tokenizer = None
    finbert_model = None

def get_sentiment_score(text):
    """Analyze sentiment using FinBERT. Returns score from -1 to 1."""
    if not TRANSFORMERS_AVAILABLE or tokenizer is None or finbert_model is None:
        return 0.0
    
    if not text or len(text.strip()) == 0:
        return 0.0
    try:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True, max_length=512)
        with torch.no_grad():
            outputs = finbert_model(**inputs)
            predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
        positive, negative, neutral = predictions[0]
        return (positive - negative).item()
    except Exception as e:
        print(f"Sentiment analysis error: {e}")
        return 0.0

def fetch_yahoo_news(symbol, max_articles=30):
    """Fetch news from Yahoo Finance (free, no API key)"""
    try:
        ticker = yf.Ticker(symbol)
        news = ticker.news
        if news:
            articles = []
            for item in news[:max_articles]:
                articles.append({
                    'title': item.get('title', ''),
                    'description': item.get('summary', ''),
                    'publishedAt': datetime.fromtimestamp(item.get('providerPublishTime', 0)).isoformat()
                })
            print(f"✓ Fetched {len(articles)} articles from Yahoo Finance")
            return articles
        return []
    except Exception as e:
        print(f"Yahoo Finance error: {e}")
        return []

def fetch_newsapi(symbol, company_name, days_back=7, target_date=None):
    """Fetch from NewsAPI.org - requires free API key"""
    api_key = config["sentiment"]["news_api_key"]
    if api_key == "YOUR_API_KEY_HERE":
        print("⚠ Set your NewsAPI key in config. Get free key: https://newsapi.org/")
        return []
    
    # Use target_date if provided, otherwise use current date
    end_date = target_date if target_date else datetime.now()
    start_date = end_date - timedelta(days=days_back)
    
    url = "https://newsapi.org/v2/everything"
    params = {
        'q': f"{company_name} OR {symbol}",
        'from': start_date.strftime('%Y-%m-%d'),
        'to': end_date.strftime('%Y-%m-%d'),
        'language': 'en',
        'sortBy': 'relevancy',
        'apiKey': api_key,
        'pageSize': 50
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        if data['status'] == 'ok':
            print(f"✓ Fetched {len(data.get('articles', []))} from NewsAPI")
            return data.get('articles', [])
        return []
    except Exception as e:
        print(f"NewsAPI error: {e}")
        return []

def get_news_sentiment(symbol, company_name, target_date=None):
    """Get recent news sentiment"""
    date_str = target_date.strftime('%Y-%m-%d') if target_date else 'current'
    print(f"\n{'='*70}\nFetching news for {symbol} (as of {date_str})\n{'='*70}")
    
    if not TRANSFORMERS_AVAILABLE:
        print("⚠ Sentiment analysis disabled (transformers not installed)")
        print("  Install with: pip install transformers torch")
        return 0.0, []
    
    # Try NewsAPI first, fallback to Yahoo
    articles = fetch_newsapi(symbol, company_name, target_date=target_date) if config["sentiment"]["use_real_api"] else []
    if not articles:
        articles = fetch_yahoo_news(symbol)
    
    if not articles:
        print("⚠ No articles found. Using neutral sentiment.")
        return 0.0, []
    
    # Analyze sentiment
    sentiments = []
    print(f"\nAnalyzing {len(articles)} articles with FinBERT...\n{'-'*70}")
    
    for i, article in enumerate(articles[:25]):
        title = article.get('title', '')
        desc = article.get('description', '')
        text = f"{title}. {desc}" if desc else title
        
        if text:
            score = get_sentiment_score(text)
            sentiments.append(score)
            
            if i < 5:  # Show first 5
                label = "POS" if score > 0.1 else "NEG" if score < -0.1 else "NEU"
                print(f"{i+1}. [{label}] {score:+.3f} | {title[:60]}...")
    
    if sentiments:
        avg = np.mean(sentiments)
        pos_pct = sum(1 for s in sentiments if s > 0.1) / len(sentiments) * 100
        neg_pct = sum(1 for s in sentiments if s < -0.1) / len(sentiments) * 100
        print(f"{'-'*70}")
        print(f"📊 Average: {avg:+.3f} | Positive: {pos_pct:.0f}% | Negative: {neg_pct:.0f}%")
        print(f"{'='*70}\n")
        return avg, sentiments
    return 0.0, []

# Get ticker info
ticker_obj = yf.Ticker(config["data"]["symbol"])
try:
    company_name = ticker_obj.info.get('longName', config["data"]["symbol"])
except:
    company_name = config["data"]["symbol"]
print(f"📈 Analyzing: {company_name} ({config['data']['symbol']})")

# Download price data
data = ticker_obj.history(period=config["data"]["period"])

# Filter data up to target date
target_date = config["data"]["target_date"]
data = data[data.index <= target_date]

if len(data) == 0:
    print(f"❌ Error: No data available up to {target_date.strftime('%Y-%m-%d')}")
    sys.exit(1)

data_date = [date.strftime('%Y-%m-%d') for date in data.index]
data_close_price = data['Close'].values
num_data_points = len(data_date)
print(f"Data points: {num_data_points} from {data_date[0]} to {data_date[-1]}")
print(f"Training model as if today is: {data_date[-1]}")

# Get sentiment
current_sentiment, _ = get_news_sentiment(config["data"]["symbol"], company_name, config["data"]["target_date"])

# Create sentiment time series (with realistic variation)
baseline = current_sentiment
trend = np.linspace(-0.1, 0.1, num_data_points)
noise = np.random.normal(0, 0.15, num_data_points)
seasonal = 0.1 * np.sin(np.linspace(0, 4*np.pi, num_data_points))
sentiment_data = np.clip(baseline + trend + noise + seasonal, -1, 1)

# Normalizers
class Normalizer():
    def __init__(self):
        self.mu = self.sd = None
    def fit_transform(self, x):
        self.mu = np.mean(x, axis=0, keepdims=True)
        self.sd = np.std(x, axis=0, keepdims=True)
        return (x - self.mu) / self.sd
    def inverse_transform(self, x):
        return (x * self.sd) + self.mu

price_scaler = Normalizer()
sentiment_scaler = Normalizer()
normalized_price = price_scaler.fit_transform(data_close_price)
normalized_sentiment = sentiment_scaler.fit_transform(sentiment_data)
normalized_data = np.column_stack([normalized_price, normalized_sentiment])

# Prepare windowed data
def prepare_data_x(x, ws):
    n = x.shape[0] - ws + 1
    out = np.lib.stride_tricks.as_strided(x, shape=(n, ws, x.shape[1]), 
                                          strides=(x.strides[0], x.strides[0], x.strides[1]))
    return out[:-1], out[-1]

def prepare_data_y(x, ws):
    return x[ws:, 0]

data_x, data_x_unseen = prepare_data_x(normalized_data, config["data"]["window_size"])
data_y = prepare_data_y(normalized_data, config["data"]["window_size"])

# Split
split_index = int(len(data_y) * config["data"]["train_split_size"])
data_x_train, data_x_val = data_x[:split_index], data_x[split_index:]
data_y_train, data_y_val = data_y[:split_index], data_y[split_index:]

# Dataset
class TimeSeriesDataset(Dataset):
    def __init__(self, x, y):
        self.x = x.astype(np.float32)
        self.y = y.astype(np.float32)
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

# Model
class LSTMModel(nn.Module):
    def __init__(self, input_size=2, hidden_layer_size=32, num_layers=2, output_size=1, dropout=0.2):
        super().__init__()
        self.hidden_layer_size = hidden_layer_size
        self.linear_1 = nn.Linear(input_size, hidden_layer_size)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(hidden_layer_size, hidden_size=hidden_layer_size, 
                           num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.linear_2 = nn.Linear(num_layers * hidden_layer_size, output_size)
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
        bs = x.shape[0]
        x = self.relu(self.linear_1(x))
        _, (h_n, _) = self.lstm(x)
        x = h_n.permute(1, 0, 2).reshape(bs, -1)
        x = self.dropout(x)
        return self.linear_2(x)[:, -1]

def run_epoch(dataloader, is_training=False):
    epoch_loss = 0
    model.train() if is_training else model.eval()
    
    for x, y in dataloader:
        if is_training:
            optimizer.zero_grad()
        x, y = x.to(config["training"]["device"]), y.to(config["training"]["device"])
        out = model(x)
        loss = criterion(out, y)
        if is_training:
            loss.backward()
            optimizer.step()
        epoch_loss += loss.detach().item() / x.shape[0]
    
    return epoch_loss

# Train
model = LSTMModel(input_size=2, hidden_layer_size=config["model"]["lstm_size"],
                  num_layers=config["model"]["num_lstm_layers"], dropout=config["model"]["dropout"])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=config["training"]["learning_rate"], betas=(0.9, 0.98), eps=1e-9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=config["training"]["scheduler_step_size"], gamma=0.1)

print("\n🚀 Training with price + sentiment...")
for epoch in range(config["training"]["num_epoch"]):
    loss_train = run_epoch(train_dataloader, True)
    loss_val = run_epoch(val_dataloader)
    scheduler.step()
    if epoch % 10 == 0:
        print(f'Epoch[{epoch+1}/{config["training"]["num_epoch"]}] | train:{loss_train:.6f}, val:{loss_val:.6f}')

print("Model training complete")

# Predictions on training/validation data
model.eval()
train_dataloader = DataLoader(dataset_train, batch_size=config["training"]["batch_size"], shuffle=False)
val_dataloader = DataLoader(dataset_val, batch_size=config["training"]["batch_size"], shuffle=False)

predicted_train = np.concatenate([model(x.to(config["training"]["device"])).cpu().detach().numpy() 
                                 for x, _ in train_dataloader])
predicted_val = np.concatenate([model(x.to(config["training"]["device"])).cpu().detach().numpy() 
                               for x, _ in val_dataloader])

# Save predicted vs actual validation data to CSV
print("\nSaving predicted vs actual validation data to CSV...")
validation_data = []
val_start_index = split_index + config["data"]["window_size"]

for i in range(len(predicted_val)):
    validation_data.append({
        'Date': data_date[val_start_index + i],
        'Actual_Price': float(price_scaler.inverse_transform(np.array([[data_y_val[i]]]))[0][0]),
        'Predicted_Price': float(price_scaler.inverse_transform(np.array([[predicted_val[i]]]))[0][0])
    })

validation_df = pd.DataFrame(validation_data)
validation_df.to_csv('predicted_vs_actual_validation.csv', index=False)
print("✓ Saved to: predicted_vs_actual_validation.csv")

# Predict direction for next month
print("\n🔮 Generating next month prediction...")
model.eval()

current_window = data_x_unseen.copy()
predicted_prices = []

trading_days_in_month = 22

# Use the last sentiment value and maintain it for future predictions
last_sentiment = normalized_sentiment[-1]

for day in range(trading_days_in_month):
    x = torch.tensor(current_window).float().to(config["training"]["device"]).unsqueeze(0)
    with torch.no_grad():
        predicted_normalized = model(x)
    
    predicted_normalized_np = predicted_normalized.cpu().detach().numpy()
    if predicted_normalized_np.ndim == 0:
        predicted_normalized_value = float(predicted_normalized_np)
    else:
        predicted_normalized_value = float(predicted_normalized_np[0])
    
    predicted_price = price_scaler.inverse_transform(np.array([[predicted_normalized_value]]))[0][0]
    predicted_prices.append(predicted_price)
    
    # Update window with new prediction and maintain sentiment
    new_row = np.array([predicted_normalized_value, last_sentiment])
    current_window = np.vstack([current_window[1:], new_row])

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
    'Predicted_Price': predicted_prices
})
prediction_df.to_csv('next_month_prediction.csv', index=False)
print("✓ Saved to: next_month_prediction.csv")

direction_df = pd.DataFrame({
    'Direction': [direction]
})
direction_df.to_csv('stock_direction_prediction.csv', index=False)
print(f"✓ Saved direction prediction to: stock_direction_prediction.csv")
print(f"📊 Direction: {direction}")