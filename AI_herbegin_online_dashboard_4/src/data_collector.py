"""
Data collectors voor koersen (yfinance) en optioneel Finnhub + nieuws/sentiment.
"""
import datetime as dt
from typing import Optional, Tuple
import pandas as pd
import yfinance as yf
import requests
from .utils import ensure_dir
from config import FINNHUB_KEY, DATA_DIR

def fetch_yf(ticker: str, start: str, end: str, interval: str="1d") -> pd.DataFrame:
    df = yf.download(ticker, start=start, end=end, interval=interval, auto_adjust=True, progress=False)
    if not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    df = df.rename(columns={"Adj Close": "adj_close", "Close": "close", "Open":"open", "High":"high", "Low":"low", "Volume":"volume"})
    return df

def fetch_finnhub(ticker: str, start: str, end: str) -> pd.DataFrame:
    if not FINNHUB_KEY:
        return pd.DataFrame()
    # Finnhub candles (daily)
    # convert to unix timestamps
    start_ts = int(pd.Timestamp(start).timestamp())
    end_ts = int(pd.Timestamp(end).timestamp())
    url = f"https://finnhub.io/api/v1/stock/candle?symbol={ticker}&resolution=D&from={start_ts}&to={end_ts}&token={FINNHUB_KEY}"
    try:
        r = requests.get(url, timeout=15)
        js = r.json()
        if js.get("s") != "ok":
            return pd.DataFrame()
        df = pd.DataFrame(
            {
                "open": js["o"],
                "high": js["h"],
                "low": js["l"],
                "close": js["c"],
                "volume": js["v"],
            },
            index=pd.to_datetime(js["t"], unit="s"),
        )
        df.index.name = "Date"
        return df
    except Exception:
        return pd.DataFrame()

def get_prices(ticker: str, years: int=10, interval: str="1d") -> pd.DataFrame:
    end = dt.date.today()
    start = end - dt.timedelta(days=365*years)
    yf_df = fetch_yf(ticker, start.isoformat(), end.isoformat(), interval=interval)
    fh_df = fetch_finnhub(ticker, start.isoformat(), end.isoformat())
    if not fh_df.empty:
        # voorkeur aan yfinance, vul ontbrekende gaten met Finnhub
        yf_df = yf_df.combine_first(fh_df)
    return yf_df

def cache_prices(ticker: str, df: pd.DataFrame) -> str:
    ensure_dir(f"{DATA_DIR}/prices")
    path = f"{DATA_DIR}/prices/{ticker}_{df.index.min().date()}_{df.index.max().date()}.parquet"
    df.to_parquet(path)
    return path


# ✅ NIEUW: nieuws ophalen en omzetten naar daily sentiment
def get_daily_sentiment(ticker: str, start: str=None, end: str=None) -> pd.DataFrame:
    """
    Haalt nieuws van Finnhub en berekent daggemiddelde sentiment.
    Returned altijd een DataFrame met kolommen ['date', 'sentiment'].
    """
    if not FINNHUB_KEY:
        return pd.DataFrame(columns=["date", "sentiment"])

    if end is None:
        end = dt.date.today().isoformat()
    if start is None:
        start = (dt.date.today() - dt.timedelta(days=365)).isoformat()

    url = f"https://finnhub.io/api/v1/company-news?symbol={ticker}&from={start}&to={end}&token={FINNHUB_KEY}"
    try:
        r = requests.get(url, timeout=15)
        news = r.json()
        if not isinstance(news, list) or len(news) == 0:
            return pd.DataFrame(columns=["date", "sentiment"])

        df = pd.DataFrame(news)

        # Zorg dat er altijd een date-kolom is
        if "datetime" in df.columns:
            df["date"] = pd.to_datetime(df["datetime"], unit="s").dt.date
        elif "publishedDate" in df.columns:
            df["date"] = pd.to_datetime(df["publishedDate"]).dt.date
        else:
            return pd.DataFrame(columns=["date", "sentiment"])

        # Placeholder sentiment (als je al een model gebruikt, vervang dit!)
        df["sentiment"] = 0.0  
        if "headline" in df.columns:
            df.loc[df["headline"].str.contains("up", case=False, na=False), "sentiment"] = 1.0
            df.loc[df["headline"].str.contains("down", case=False, na=False), "sentiment"] = -1.0

        # Gemiddelde per dag
        daily_sent = df.groupby("date")["sentiment"].mean().reset_index()

        return daily_sent

    except Exception:
        return pd.DataFrame(columns=["date", "sentiment"])
