"""SwingAnalyzer — 4h price structure analysis.

Uses EMA alignment (21/50/200) and recent swing high/low structure
to determine directional bias. This is the primary gate for spot trades.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

_log = logging.getLogger(__name__)

_MIN_BARS = 50  # Need at least 50 bars for EMA-50 to be meaningful


class SwingAnalyzer:
    """Analyses 4h price structure to determine swing bias and key levels.

    Uses EMA 21/50/200 alignment as the primary bias signal. Falls back
    to neutral with no allowed directions when data is insufficient.
    """

    def analyze(self, df: pd.DataFrame) -> dict[str, Any]:
        """Analyse price structure and return swing context components.

        Args:
            df: 4h OHLCV DataFrame with at least 50 rows. Columns: open, high, low, close, volume.

        Returns:
            Dict with keys: swing_bias, allowed_directions, key_levels, confidence.
        """
        if len(df) < _MIN_BARS:
            _log.debug("Insufficient data for swing analysis (%d bars)", len(df))
            return {
                "swing_bias": "neutral",
                "allowed_directions": [],
                "key_levels": {"support": 0.0, "resistance": 0.0, "poc": 0.0},
                "confidence": 0.0,
            }

        close = df["close"]

        # EMA alignment — primary structural signal
        ema21 = close.ewm(span=21, adjust=False).mean()
        ema50 = close.ewm(span=50, adjust=False).mean()
        ema200 = close.ewm(span=200, adjust=False).mean() if len(df) >= 200 else None

        latest = close.iloc[-1]
        e21 = ema21.iloc[-1]
        e50 = ema50.iloc[-1]
        e200 = ema200.iloc[-1] if ema200 is not None else None

        # Count bullish/bearish EMA conditions (max 3)
        bullish_count = sum(
            [
                latest > e21,
                e21 > e50,
                e200 is not None and e50 > e200,
            ]
        )
        bearish_count = sum(
            [
                latest < e21,
                e21 < e50,
                e200 is not None and e50 < e200,
            ]
        )

        if bullish_count >= 2 and bullish_count > bearish_count:
            bias = "bullish"
            allowed = ["long"]
            confidence = 0.5 + (bullish_count / 3) * 0.4
        elif bearish_count >= 2 and bearish_count > bullish_count:
            bias = "bearish"
            allowed = ["short"]
            confidence = 0.5 + (bearish_count / 3) * 0.4
        elif bullish_count > bearish_count:
            # Weak bullish structure — 1-of-3 EMA conditions met (e.g. close > EMA21)
            bias = "bullish"
            allowed = ["long"]
            confidence = 0.60
        elif bearish_count > bullish_count:
            # Weak bearish structure — 1-of-3 EMA conditions met
            bias = "bearish"
            allowed = ["short"]
            confidence = 0.60
        else:
            # Truly neutral — EMA counts tied (0-0 or 1-1); allow both directions
            # with reduced confidence so quality signals still have to clear threshold
            bias = "neutral"
            allowed = ["long", "short"]
            confidence = 0.45

        # Key levels: recent 20-bar swing high/low as resistance/support
        window = min(20, len(df))
        recent = df.tail(window)
        support = float(recent["low"].min())
        resistance = float(recent["high"].max())
        poc = float(recent["close"].median())

        key_levels: dict[str, float] = {
            "support": support,
            "resistance": resistance,
            "poc": poc,
        }

        # Prev-day high/low — only available when df has a DatetimeIndex
        try:
            dates = pd.DatetimeIndex(df.index).normalize()
            unique_dates = sorted(dates.unique())
            if len(unique_dates) >= 2:
                yesterday = unique_dates[-2]
                mask = dates == yesterday
                key_levels["pdh"] = float(df.loc[mask, "high"].max())
                key_levels["pdl"] = float(df.loc[mask, "low"].min())
        except Exception:
            pass  # Integer index or other non-datetime index — pdh/pdl omitted

        return {
            "swing_bias": bias,
            "allowed_directions": allowed,
            "key_levels": key_levels,
            "confidence": round(confidence, 3),
        }
