import os
import sys
import csv
from datetime import datetime, timedelta, timezone
from typing import List, Dict

import requests
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# =========================
# CONFIG
# =========================

NEWS_API_URL = "https://newsapi.org/v2/everything"
FINBERT_MODEL = "ProsusAI/finbert"
MAX_TOKENS = 512
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

LABELS = ["negative", "neutral", "positive"]

# =========================
# LOAD FINBERT
# =========================

tokenizer = AutoTokenizer.from_pretrained(FINBERT_MODEL)
model = AutoModelForSequenceClassification.from_pretrained(FINBERT_MODEL)
model.to(DEVICE)
model.eval()

# =========================
# NEWS FETCHING
# =========================

def fetch_news(ticker: str, days: int = 7) -> List[Dict]:
    api_key = "f18c946706924f6f8f864648b5338afd"
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set")

    from_date = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")

    params = {
        "q": ticker,
        "from": from_date,
        "language": "en",
        "sortBy": "relevancy",
        "pageSize": 20,
        "apiKey": api_key,
    }

    response = requests.get(NEWS_API_URL, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()

    articles = []
    for a in data.get("articles", []):
        if a.get("title") and a.get("description"):
            articles.append({
                "title": a["title"],
                "description": a["description"],
                "source": a["source"]["name"],
                "url": a["url"]
            })

    return articles

# =========================
# SENTIMENT ANALYSIS
# =========================

def finbert_sentiment(text: str) -> Dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=MAX_TOKENS
    ).to(DEVICE)

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = F.softmax(logits, dim=1)[0]

    return {
        "label": LABELS[probs.argmax().item()],
        "scores": {LABELS[i]: probs[i].item() for i in range(3)}
    }

def analyze_articles(articles: List[Dict]) -> List[Dict]:
    results = []
    for a in articles:
        text = f"{a['title']}. {a['description']}"
        sentiment = finbert_sentiment(text)

        results.append({
            **a,
            "sentiment": sentiment["label"],
            "scores": sentiment["scores"]
        })
    return results

# =========================
# SUMMARY + RANKING
# =========================

def summarize_percentages(results: List[Dict]) -> Dict:
    total = len(results)
    if total == 0:
        return {"positive_pct": 0.0, "negative_pct": 0.0, "neutral_pct": 0.0}

    pos = sum(1 for r in results if r["sentiment"] == "positive")
    neg = sum(1 for r in results if r["sentiment"] == "negative")
    neu = sum(1 for r in results if r["sentiment"] == "neutral")

    return {
        "positive_pct": round(pos / total * 100, 1),
        "negative_pct": round(neg / total * 100, 1),
        "neutral_pct": round(neu / total * 100, 1),
    }

def get_top_articles(results: List[Dict], top_n: int = 5) -> List[Dict]:
    return sorted(
        results,
        key=lambda r: r["scores"]["positive"],
        reverse=True
    )[:top_n]

# =========================
# CSV EXPORT
# =========================

def save_results_to_csv(ticker: str, results: Dict, output_dir: str = "outputs") -> str:
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"{ticker}_sentiment_{timestamp}.csv"
    filepath = os.path.join(output_dir, filename)

    fieldnames = [
        "ticker",
        "timestamp",
        "title",
        "source",
        "url",
        "sentiment",
        "positive_score",
        "neutral_score",
        "negative_score",
    ]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for a in results["all_articles"]:
            writer.writerow({
                "ticker": ticker,
                "timestamp": timestamp,
                "title": a["title"],
                "source": a["source"],
                "url": a["url"],
                "sentiment": a["sentiment"],
                "positive_score": round(a["scores"]["positive"], 4),
                "neutral_score": round(a["scores"]["neutral"], 4),
                "negative_score": round(a["scores"]["negative"], 4),
            })

    return filepath

# =========================
# ENTRY POINT
# =========================

def run_analysis(ticker: str) -> Dict:
    articles = fetch_news(ticker)
    analyzed = analyze_articles(articles)

    return {
        "ticker": ticker,
        "overall_percentages": summarize_percentages(analyzed),
        "top_articles": get_top_articles(analyzed),
        "all_articles": analyzed
    }

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "IBM"
    result = run_analysis(ticker)

    csv_path = save_results_to_csv(ticker, result)

    print("\nOverall Sentiment (%)")
    print("=====================")
    print(result["overall_percentages"])

    print("\nTop Articles")
    print("============")
    for a in result["top_articles"]:
        print(f"[{a['sentiment'].upper()}] {a['title']}")

    print(f"\nCSV saved to: {csv_path}")