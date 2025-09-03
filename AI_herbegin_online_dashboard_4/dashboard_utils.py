
import os
import json
from pathlib import Path
from typing import Optional, Tuple, List
import pandas as pd
import numpy as np

def find_best_model(models_dir: str) -> Optional[str]:
    """
    Zoek het meest 'waarschijnlijke' RL-model in de map.
    We kiezen:
      1) een bestand met 'best' in de naam (liefst de laatste volgens timestamp)
      2) anders het meest recent gewijzigde .zip-bestand
    Retourneert pad (str) of None.
    """
    p = Path(models_dir)
    if not p.exists():
        return None
    cands = sorted(p.glob("*.zip"), key=lambda x: x.stat().st_mtime, reverse=True)
    if not cands:
        return None
    best_like = [x for x in cands if "best" in x.name.lower()]
    if best_like:
        # kies meest recente "best*"
        best_like = sorted(best_like, key=lambda x: x.stat().st_mtime, reverse=True)
        return str(best_like[0])
    return str(cands[0])


def ta_features(df: pd.DataFrame) -> pd.DataFrame:
    """Bereken technische indicatoren (returns, SMA, EMA, RSI, MACD, ATR) uit een OHLCV-dataset."""
    if df is None or df.empty:
        return pd.DataFrame()

    df = df.copy()

    # ✅ Flatten kolomnamen
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0].lower() for c in df.columns]
    else:
        df.columns = [str(c).lower() for c in df.columns]

    # Normaliseer kolomnamen
    rename_map = {
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "adj close": "close",
        "volume": "volume",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    required = ["open", "high", "low", "close", "volume"]
    if not all(x in df.columns for x in required):
        raise ValueError(f"Data mist verplichte OHLCV-kolommen: {df.columns}")

    out = df.copy()

    out["ret_1d"] = out["close"].pct_change(1)
    out["ret_5d"] = out["close"].pct_change(5)
    out["sma_10"] = out["close"].rolling(10).mean()
    out["sma_50"] = out["close"].rolling(50).mean()
    out["ema_10"] = out["close"].ewm(span=10, adjust=False).mean()
    out["ema_20"] = out["close"].ewm(span=20, adjust=False).mean()

    delta = out["close"].diff()
    up = delta.clip(lower=0).rolling(14).mean()
    down = (-delta.clip(upper=0)).rolling(14).mean()
    rs = up / down.replace(0, np.nan)
    out["rsi_14"] = 100 - (100 / (1 + rs))

    ema12 = out["close"].ewm(span=12, adjust=False).mean()
    ema26 = out["close"].ewm(span=26, adjust=False).mean()
    out["macd"] = ema12 - ema26
    out["macd_sig"] = out["macd"].ewm(span=9, adjust=False).mean()

    tr1 = (out["high"] - out["low"]).abs()
    tr2 = (out["high"] - out["close"].shift()).abs()
    tr3 = (out["low"] - out["close"].shift()).abs()
    out["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    return out
, metrics
