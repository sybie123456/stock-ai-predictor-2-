"""
Feature engineering: technische indicatoren + sentiment merge.
"""
import pandas as pd
import numpy as np
import ta

"""def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if df.empty: 
        return df
    df["ret"] = df["close"].pct_change().fillna(0.0)
    df["sma_10"] = df["close"].rolling(10).mean()
    df["sma_50"] = df["close"].rolling(50).mean()
    df["rsi_14"] = ta.momentum.rsi(df["close"], window=14)
    df["macd"] = ta.trend.macd_diff(df["close"])
    df = df.fillna(method="bfill").fillna(method="ffill")
    return df"""
def _ensure_series(x):
    """Zorgt dat input altijd een 1D pandas Series is."""
    if isinstance(x, pd.DataFrame):
        if x.shape[1] == 1:
            return x.iloc[:, 0]
        else:
            raise ValueError("DataFrame heeft meer dan 1 kolom, specificeer welke kolom.")
    elif isinstance(x, pd.Series):
        return x
    else:
        # bijvoorbeeld numpy array
        return pd.Series(x).squeeze()


def add_tech_indicators(df: pd.DataFrame) -> pd.DataFrame:
    close = _ensure_series(df["close"])
    volume = _ensure_series(df["volume"]) if "volume" in df.columns else None

    # Simple Moving Averages
    df["sma_20"] = ta.trend.sma_indicator(close, window=20)
    df["sma_50"] = ta.trend.sma_indicator(close, window=50)

    # RSI
    df["rsi_14"] = ta.momentum.rsi(close, window=14)

    # MACD
    df["macd"] = ta.trend.macd(close)
    df["macd_signal"] = ta.trend.macd_signal(close)

    # eventueel nog meer indicatoren mogelijk
    if volume is not None:
        df["obv"] = ta.volume.on_balance_volume(close, volume)

    return df

def merge_with_sentiment(price_df: pd.DataFrame, daily_sent: pd.DataFrame) -> pd.DataFrame:
    """
    Merge stock prices met daily sentiment.
    Verwacht dat 'daily_sent' een kolom 'date' heeft of een datetime index.
    """
    if daily_sent is None or daily_sent.empty:
        price_df["sentiment"] = 0.0
        return price_df

    # Zorg dat er altijd een 'date'-kolom is
    if "date" in daily_sent.columns:
        sent = daily_sent.copy()
        sent["date"] = pd.to_datetime(sent["date"])
        sent = sent.set_index("date")
    else:
        # fallback: misschien is het al een DatetimeIndex
        if isinstance(daily_sent.index, pd.DatetimeIndex):
            sent = daily_sent.copy()
            sent.index.name = "date"
        else:
            raise ValueError("daily_sent moet een 'date'-kolom of datetime index hebben.")

    # resample naar dagelijks gemiddelde (voor het geval er meerdere waarden per dag zijn)
    sent = sent.resample("D").mean().fillna(0.0)

    # Prices indexeren op datetime
    df = price_df.copy()
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # merge
    df = df.join(sent, how="left")
    df["sentiment"] = df["sentiment"].fillna(0.0)

    return df

def make_observation_matrix(df: pd.DataFrame, feature_cols=None):
    if feature_cols is None:
        feature_cols = ["ret","sma_10","sma_50","rsi_14","macd","sentiment","volume"]
    X = df[feature_cols].astype(float).values
    return X, feature_cols
