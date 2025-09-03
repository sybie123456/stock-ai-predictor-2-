"""
Nieuwsbronnen: NewsAPI.org + Finnhub company news.
"""
from typing import List, Dict
import datetime as dt
import requests
import pandas as pd
from config import NEWSAPI_KEY, FINNHUB_KEY

def fetch_newsapi(query: str, from_date: str, to_date: str, language="en", page_size=100) -> pd.DataFrame:
    if not NEWSAPI_KEY:
        return pd.DataFrame()
    url = "https://newsapi.org/v2/everything"
    params = {"q": query, "from": from_date, "to": to_date, "language": language, "pageSize": page_size, "apiKey": NEWSAPI_KEY, "sortBy":"relevancy"}
    try:
        r = requests.get(url, params=params, timeout=15)
        data = r.json()
        arts = data.get("articles", [])
        rows = []
        for a in arts:
            rows.append({"date": a["publishedAt"], "title": a["title"], "source": a["source"]["name"], "url": a["url"], "content": a.get("content") or a.get("description")})
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"]).dt.tz_convert(None)
            df = df.sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()

def fetch_finnhub_company_news(ticker: str, from_date: str, to_date: str) -> pd.DataFrame:
    if not FINNHUB_KEY:
        return pd.DataFrame()
    url = "https://finnhub.io/api/v1/company-news"
    params = {"symbol": ticker, "from": from_date, "to": to_date, "token": FINNHUB_KEY}
    try:
        r = requests.get(url, params=params, timeout=15)
        arr = r.json()
        rows = [{"date": x.get("datetime"), "title": x.get("headline"), "source": x.get("source"), "url": x.get("url"), "content": x.get("summary")} for x in arr]
        df = pd.DataFrame(rows)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"], unit="s").dt.tz_convert(None)
            df = df.sort_values("date")
        return df
    except Exception:
        return pd.DataFrame()

def get_news(ticker_or_query: str, from_date: str, to_date: str) -> pd.DataFrame:
    a = fetch_newsapi(ticker_or_query, from_date, to_date)
    b = fetch_finnhub_company_news(ticker_or_query, from_date, to_date)
    if a.empty and b.empty:
        return pd.DataFrame(columns=["date","title","source","url","content"])
    if a.empty: return b
    if b.empty: return a
    df = pd.concat([a,b], ignore_index=True).drop_duplicates(subset=["title","url"]).sort_values("date")
    return df
