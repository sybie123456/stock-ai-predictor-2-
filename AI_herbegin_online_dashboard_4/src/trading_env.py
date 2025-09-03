"""
Gymnasium-achtige omgeving voor mixed replay: sample historische of recente vensters.
Acties: 0 = houden, 1 = kopen (long), 2 = verkopen (flat/close).
Eenvoudige PnL beloning.
"""
import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces
import random

class MixedTradingEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, df: pd.DataFrame, min_len=64, max_len=512, mixed_ratio=0.5):
        super().__init__()
        self.df_full = df
        self.min_len = min_len
        self.max_len = max_len
        self.mixed_ratio = mixed_ratio
        self.position = 0  # 0 flat, 1 long
        self.entry_price = 0.0
        self.t = 0

        # ✅ alleen features (zonder "close")
        feature_cols = [c for c in df.columns if c != "close"]
        self.feature_cols = feature_cols
        obs_dim = len(feature_cols)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32
        )
        self.action_space = spaces.Discrete(3)

    def _sample_window(self):
        L = random.randint(self.min_len, self.max_len)
        n = len(self.df_full)
        if n <= L:
            start = 0
        else:
            if random.random() < self.mixed_ratio:
                start = random.randint(0, max(0, n - L))
            else:
                recent_start = max(0, int(n * 0.75) - L)
                start = random.randint(recent_start, max(recent_start, n - L))
        end = start + L
        return self.df_full.iloc[start:end].dropna()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.window = self._sample_window()
        self.prices = self.window["close"].values
        self.features = self.window[self.feature_cols].values  # ✅ enkel features
        self.t = 0
        self.position = 0
        self.entry_price = 0.0
        return self.features[self.t].astype(np.float32), {}

    def step(self, action):
        reward = 0.0
        price = self.prices[self.t]
        terminated = False
        truncated = False

        # actie effect
        if action == 1:  # long openen/houden
            if self.position == 0:
                self.position = 1
                self.entry_price = price
        elif action == 2:  # sluiten
            if self.position == 1:
                reward = (price - self.entry_price) / self.entry_price
                self.position = 0
                self.entry_price = 0.0

        self.t += 1
        if self.t >= len(self.prices) - 1:
            terminated = True
            if self.position == 1:
                price = self.prices[self.t]
                reward += (price - self.entry_price) / self.entry_price
                self.position = 0
                self.entry_price = 0.0

        obs = self.features[min(self.t, len(self.features) - 1)].astype(np.float32)
        return obs, float(reward), terminated, truncated, {}

