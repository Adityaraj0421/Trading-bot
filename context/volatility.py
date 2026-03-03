"""VolatilityAnalyzer — classify current ATR vs rolling baseline."""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

_ATR_PERIOD = 14
_VOL_LOOKBACK = 30
_MIN_BARS = _ATR_PERIOD + _VOL_LOOKBACK  # 44


class VolatilityAnalyzer:
    """Computes 14-bar ATR% and classifies vs 30-bar rolling mean.

    Returns one of: "low", "normal", "elevated", "extreme".
    Falls back to "normal" on insufficient data.
    """

    def analyze(self, df: pd.DataFrame | None) -> dict[str, Any]:
        """Classify volatility regime from 1h OHLCV data.

        Args:
            df: 1h OHLCV DataFrame. Needs ≥44 rows. None → "normal".

        Returns:
            Dict with key ``volatility_regime``.
        """
        if df is None or len(df) < _MIN_BARS:
            return {"volatility_regime": "normal"}

        high = df["high"]
        low = df["low"]
        close = df["close"]

        prev_close = close.shift(1)
        tr = pd.concat(
            [
                (high - low),
                (high - prev_close).abs(),
                (low - prev_close).abs(),
            ],
            axis=1,
        ).max(axis=1)

        atr = tr.rolling(_ATR_PERIOD).mean()
        atr_pct = atr / close

        rolling_mean = atr_pct.rolling(_VOL_LOOKBACK).mean()

        current = atr_pct.iloc[-1]
        baseline = rolling_mean.iloc[-1]

        if pd.isna(current) or pd.isna(baseline) or baseline == 0:
            return {"volatility_regime": "normal"}

        ratio = current / baseline

        if ratio < 0.5:
            regime = "low"
        elif ratio < 1.5:
            regime = "normal"
        elif ratio < 2.5:
            regime = "elevated"
        else:
            regime = "extreme"

        _log.debug("VolatilityAnalyzer: ATR%% ratio=%.2f → %s", ratio, regime)
        return {"volatility_regime": regime}
