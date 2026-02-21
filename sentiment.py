"""
Sentiment Analysis Module — OPTIMIZED v2.
Bounded history, reuses pre-computed indicators, cached API results.
"""

import logging
import numpy as np
import pandas as pd
import requests
from dataclasses import dataclass
from enum import Enum
from collections import deque
import time

_log = logging.getLogger(__name__)


class SentimentLevel(Enum):
    EXTREME_FEAR = "extreme_fear"
    FEAR = "fear"
    NEUTRAL = "neutral"
    GREED = "greed"
    EXTREME_GREED = "extreme_greed"


@dataclass
class SentimentState:
    fear_greed_index: int
    fear_greed_label: SentimentLevel
    volume_sentiment: float
    price_momentum_score: float
    composite_score: float
    contrarian_signal: str
    source: str


class SentimentAnalyzer:
    """
    OPTIMIZED: Caches Fear & Greed API results (refreshes every 5 min).
    Reuses pre-computed returns from indicators instead of recalculating.
    Bounded history via deque.
    """

    FEAR_GREED_API = "https://api.alternative.me/fng/?limit=7&format=json"
    FG_CACHE_SECONDS = 300  # Cache API result for 5 minutes

    def __init__(self) -> None:
        self.last_fg_index: int = 50
        self.fg_history: deque[int] = deque(maxlen=100)
        self._fg_cache_time: float = 0
        self._fg_cache_value: int | None = None

    def analyze(self, df: pd.DataFrame, df_ind: pd.DataFrame | None = None) -> SentimentState:
        """Run sentiment analysis. Accepts df_ind to reuse pre-computed returns."""
        fg_index, fg_label, fg_source = self._fetch_fear_greed()
        vol_sentiment = self._volume_sentiment(df)
        momentum = self._price_momentum(df, df_ind)

        fg_normalized = (fg_index - 50) / 50
        composite = fg_normalized * 0.3 + vol_sentiment * 0.35 + momentum * 0.35
        contrarian = self._contrarian_signal(fg_index, composite)

        return SentimentState(
            fear_greed_index=fg_index,
            fear_greed_label=fg_label,
            volume_sentiment=round(vol_sentiment, 4),
            price_momentum_score=round(momentum, 4),
            composite_score=round(composite, 4),
            contrarian_signal=contrarian,
            source=fg_source,
        )

    def _fetch_fear_greed(self) -> tuple[int, SentimentLevel, str]:
        """Cached Fear & Greed fetch — only hits API every 5 minutes."""
        now = time.time()
        if self._fg_cache_value is not None and (now - self._fg_cache_time) < self.FG_CACHE_SECONDS:
            v = self._fg_cache_value
            return v, self._classify_fg(v), "cached"

        try:
            resp = requests.get(self.FEAR_GREED_API, timeout=3)
            data = resp.json()
            value = int(data["data"][0]["value"])
            self.last_fg_index = value
            self.fg_history.append(value)
            self._fg_cache_value = value
            self._fg_cache_time = now
            return value, self._classify_fg(value), "api"
        except requests.RequestException as e:
            _log.warning("Fear & Greed API network error: %s", e)
            return self.last_fg_index, self._classify_fg(self.last_fg_index), "fallback"
        except (ValueError, KeyError, IndexError) as e:
            _log.error("Fear & Greed API parse error: %s", e)
            return self.last_fg_index, self._classify_fg(self.last_fg_index), "fallback"

    def _classify_fg(self, value: int) -> SentimentLevel:
        if value <= 25: return SentimentLevel.EXTREME_FEAR
        if value <= 45: return SentimentLevel.FEAR
        if value <= 55: return SentimentLevel.NEUTRAL
        if value <= 75: return SentimentLevel.GREED
        return SentimentLevel.EXTREME_GREED

    def _volume_sentiment(self, df: pd.DataFrame) -> float:
        if len(df) < 20:
            return 0.0
        recent = df.iloc[-20:]  # View, not copy
        returns = recent["close"].pct_change()
        vol_mean = recent["volume"].mean()
        if vol_mean == 0:
            return 0.0
        vol_norm = recent["volume"] / vol_mean

        obv_direction = np.sign(returns.values[1:]) * vol_norm.values[1:]
        obv_score = float(np.nanmean(obv_direction[-10:]))

        up_mask = returns.values > 0
        down_mask = returns.values < 0
        up_vol = recent["volume"].values[up_mask].mean() if up_mask.sum() > 0 else 0
        down_vol = recent["volume"].values[down_mask].mean() if down_mask.sum() > 0 else 0
        total = up_vol + down_vol
        vol_ratio = (up_vol - down_vol) / total if total > 0 else 0.0

        score = obv_score * 0.5 + vol_ratio * 0.5
        return float(np.clip(score, -1, 1))

    def _price_momentum(self, df: pd.DataFrame, df_ind: pd.DataFrame | None = None) -> float:
        """Reuse pre-computed returns from indicators if available."""
        if df_ind is not None and "returns_5" in df_ind.columns:
            ret_5 = float(df_ind["returns_5"].iloc[-1])
            ret_10 = float(df_ind["returns_10"].iloc[-1])
            # Compute longer timeframes from close
            close = df["close"]
            ret_20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
            ret_50 = float(close.iloc[-1] / close.iloc[-50] - 1) if len(close) >= 50 else 0
        else:
            if len(df) < 50:
                return 0.0
            close = df["close"]
            ret_5 = float(close.iloc[-1] / close.iloc[-5] - 1) if len(close) >= 5 else 0
            ret_10 = float(close.iloc[-1] / close.iloc[-10] - 1) if len(close) >= 10 else 0
            ret_20 = float(close.iloc[-1] / close.iloc[-20] - 1) if len(close) >= 20 else 0
            ret_50 = float(close.iloc[-1] / close.iloc[-50] - 1) if len(close) >= 50 else 0

        momentum = ret_5 * 0.4 + ret_10 * 0.3 + ret_20 * 0.2 + ret_50 * 0.1
        return float(np.clip(momentum / 0.10, -1, 1))

    def _contrarian_signal(self, fg_index: int, composite: float) -> str:
        if fg_index <= 20 and composite < -0.3: return "BUY"
        if fg_index >= 80 and composite > 0.3: return "SELL"
        return "NEUTRAL"
