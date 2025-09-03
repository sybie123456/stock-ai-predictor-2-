import os

# API keys (optioneel)
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "ce41c670-f771-4b26-af96-7bcc07df3cad")           # https://newsapi.org
FINNHUB_KEY = os.getenv("FINNHUB_KEY", "d2ljm61r01qr27gjjl0gd2ljm61r01qr27gjjl10")           # https://finnhub.io

# Data & model paden
DATA_DIR = os.getenv("DATA_DIR", "data_cache")
MODEL_DIR = os.getenv("MODEL_DIR", "models")
BEST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading_best.zip")
LAST_MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading_last.zip")

# Online leerconfig
VALIDATION_START = os.getenv("VALIDATION_START", "2018-01-01")
VALIDATION_END = os.getenv("VALIDATION_END", "2020-12-31")
HISTORY_YEARS = int(os.getenv("HISTORY_YEARS", "10"))   # hoeveel jaar historische data minimaal laden
MIXED_REPLAY_RATIO = float(os.getenv("MIXED_REPLAY_RATIO", "0.5"))  # kans om een historisch venster te sample'n
MIN_EPISODE_LENGTH = int(os.getenv("MIN_EPISODE_LENGTH", "64"))
MAX_EPISODE_LENGTH = int(os.getenv("MAX_EPISODE_LENGTH", "512"))
