import os
import sys
import datetime
import requests
from typing import List, Dict

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
# NEWS FETCHING (RAW REST)
# =========================

def fetch_news(ticker: str, days: int = 7) -> List[Dict]:
    api_key = "f18c946706924f6f8f864648b5338afd"
    if not api_key:
        raise RuntimeError("NEWSAPI_KEY not set")

    from_date = (datetime.datetime.utcnow() - datetime.timedelta(days=days)).strftime("%Y-%m-%d")

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
# SUMMARY
# =========================

def summarize_sentiment(results: List[Dict]) -> Dict:
    counts = {"positive": 0, "neutral": 0, "negative": 0}
    for r in results:
        counts[r["sentiment"]] += 1

    total = sum(counts.values())
    overall = max(counts, key=counts.get) if total else "neutral"

    return {
        "total_articles": total,
        "breakdown": counts,
        "overall_sentiment": overall
    }

# =========================
# ENTRY POINT
# =========================

def run_analysis(ticker: str) -> Dict:
    articles = fetch_news(ticker)
    analyzed = analyze_articles(articles)

    return {
        "ticker": ticker,
        "summary": summarize_sentiment(analyzed),
        "articles": analyzed
    }

if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "IBM"
    result = run_analysis(ticker)
    print(result["summary"])