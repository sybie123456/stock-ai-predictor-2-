"""
Sentiment analyse met Hugging Face pipeline + Vader fallback.
"""
from typing import List
import pandas as pd

try:
    from transformers import pipeline
    _hf = pipeline("sentiment-analysis")
except Exception:
    _hf = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:
    _vader = None

def score_text(text: str) -> float:
    if not isinstance(text, str) or not text.strip():
        return 0.0
    if _hf is not None:
        out = _hf(text[:512])[0]
        label = out["label"].lower()
        score = out["score"]
        if "pos" in label:
            return +score
        if "neg" in label:
            return -score
        return 0.0
    if _vader is not None:
        return _vader.polarity_scores(text)["compound"]
    return 0.0

def aggregate_daily_sentiment(df_news: pd.DataFrame) -> pd.DataFrame:
    if df_news.empty:
        return pd.DataFrame(columns=["sentiment"])
    df = df_news.copy()
    df["text"] = (df["title"].fillna("") + " " + df["content"].fillna("")).str.strip()
    df["s"] = df["text"].map(score_text)
    df = df.groupby(pd.to_datetime(df["date"]).dt.date)["s"].mean().to_frame("sentiment")
    df.index = pd.to_datetime(df.index)
    return df
