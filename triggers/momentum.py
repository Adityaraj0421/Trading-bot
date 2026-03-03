"""MomentumTrigger — 1h RSI + MACD zero-cross + volume confirmation.

Fires when at least 2 of these 3 conditions are true:
  - RSI crosses the 50 line (above for long, below for short)
  - MACD crosses the zero line (bullish for long, bearish for short)
  - Volume confirms (> 1.5× 20-bar average) in the correct direction
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_MIN_BARS = 26           # Need 26 bars for MACD(12,26)
_SIGNAL_TTL_MINUTES = 75  # Expires after ~1 candle + buffer
_VOL_MULT = 1.5          # Volume multiplier required for confirmation


class MomentumTrigger:
    """Generates momentum-based TriggerSignals from 1h OHLCV data.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Used to set symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]  # "BTC/USDT" → "BTC"

    def evaluate(self, df: pd.DataFrame) -> list[TriggerSignal]:
        """Evaluate 1h OHLCV data and return any momentum triggers.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            List of TriggerSignal (empty if no momentum conditions met).
        """
        if len(df) < _MIN_BARS:
            return []

        close = df["close"]
        volume = df["volume"]

        # RSI (14-period Wilder smoothing via rolling mean)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        # MACD (12/26 EMA)
        ema12 = close.ewm(span=12, adjust=False).mean()
        ema26 = close.ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26

        # Volume ratio vs 20-bar average
        vol_ma = volume.rolling(20).mean()
        vol_ratio = volume / vol_ma.replace(0, np.nan)

        if rsi.isna().iloc[-1] or macd_line.isna().iloc[-1]:
            return []

        current_rsi = rsi.iloc[-1]
        prev_rsi = rsi.iloc[-2]
        current_macd = float(macd_line.iloc[-1])
        prev_macd = float(macd_line.iloc[-2])
        current_vol_ratio = float(vol_ratio.iloc[-1]) if not vol_ratio.isna().iloc[-1] else 1.0

        # Zero-crossing conditions
        rsi_crossed_up = prev_rsi < 50 <= current_rsi
        rsi_crossed_down = prev_rsi > 50 >= current_rsi
        macd_crossed_up = prev_macd < 0 <= current_macd
        macd_crossed_down = prev_macd > 0 >= current_macd
        vol_confirmed = current_vol_ratio >= _VOL_MULT

        long_score = sum([rsi_crossed_up, macd_crossed_up, vol_confirmed and current_rsi > 50])
        short_score = sum([rsi_crossed_down, macd_crossed_down, vol_confirmed and current_rsi < 50])

        signals: list[TriggerSignal] = []
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        if long_score >= 2:
            strength = min(0.4 + long_score * 0.2, 0.95)
            reason_parts = []
            if rsi_crossed_up:
                reason_parts.append(f"RSI crossed 50↑ ({current_rsi:.1f})")
            if macd_crossed_up:
                reason_parts.append("MACD zero-cross↑")
            if vol_confirmed:
                reason_parts.append(f"vol {current_vol_ratio:.1f}×")
            signals.append(
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="momentum_1h",
                    direction="long",
                    strength=strength,
                    urgency="normal",
                    symbol_scope=self._symbol_scope,
                    reason=", ".join(reason_parts),
                    expires_at=expiry,
                    raw_data={
                        "rsi": round(current_rsi, 2),
                        "macd": round(current_macd, 4),
                        "vol_ratio": round(current_vol_ratio, 2),
                    },
                )
            )

        if short_score >= 2:
            strength = min(0.4 + short_score * 0.2, 0.95)
            reason_parts = []
            if rsi_crossed_down:
                reason_parts.append(f"RSI crossed 50↓ ({current_rsi:.1f})")
            if macd_crossed_down:
                reason_parts.append("MACD zero-cross↓")
            if vol_confirmed:
                reason_parts.append(f"vol {current_vol_ratio:.1f}×")
            signals.append(
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="momentum_1h",
                    direction="short",
                    strength=strength,
                    urgency="normal",
                    symbol_scope=self._symbol_scope,
                    reason=", ".join(reason_parts),
                    expires_at=expiry,
                    raw_data={
                        "rsi": round(current_rsi, 2),
                        "macd": round(current_macd, 4),
                        "vol_ratio": round(current_vol_ratio, 2),
                    },
                )
            )

        return signals
