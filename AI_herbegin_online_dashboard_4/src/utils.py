import os
import pandas as pd
from typing import Optional

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

def save_parquet(df: pd.DataFrame, path: str):
    ensure_dir(os.path.dirname(path))
    df.to_parquet(path, index=True)

def load_parquet(path: str) -> Optional[pd.DataFrame]:
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

def merge_on_datetime(a: pd.DataFrame, b: pd.DataFrame):
    a = a.copy()
    b = b.copy()
    a.index = pd.to_datetime(a.index)
    b.index = pd.to_datetime(b.index)
    return a.join(b, how="left").sort_index()
