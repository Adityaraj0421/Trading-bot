"""PullbackTrigger — RSI pullback + recovery detection in trending markets.

Fires when:
  - Context swing_bias is "bullish" and RSI pulls back to [42, 52] then
    starts recovering (current RSI > 3-bar low)
  - Context swing_bias is "bearish" and RSI bounces to [48, 58] then
    starts declining (current RSI < 3-bar high)
  - swing_bias == "neutral" → always returns [] (no signal in ranging markets)

Relies on the caller (TriggerEngine.on_1h_close) to pass swing_bias from
ContextEngine — avoids re-computing trend from 1h EMAs inside the trigger.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_MIN_BARS = 26                # RSI-14 needs 14+ bars; 26 adds MACD-length buffer
_SIGNAL_TTL_MINUTES = 90      # ~1.5 candle window
_RECOVERY_LOOKBACK = 3        # Bars to scan for RSI dip/bounce
_LONG_RSI_LO = 42.0           # Floor: below this is oversold, not a pullback
_LONG_RSI_HI = 52.0           # Ceiling: above this RSI hasn't pulled back to neutral
_SHORT_RSI_LO = 48.0          # Floor for short bounce zone
_SHORT_RSI_HI = 58.0          # Ceiling: above this RSI is overbought
_BASE_STRENGTH = 0.50
_MAX_STRENGTH = 0.72  # Safety cap below high-urgency perp triggers (0.75+).
                      # With current zone widths (10 RSI pts), max achievable
                      # is 0.50 + 10/14 × 0.22 ≈ 0.657 — cap is forward-looking.


class PullbackTrigger:
    """Generates pullback TriggerSignals from 1h OHLCV + context swing_bias.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Used to set symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(
        self, df: pd.DataFrame, swing_bias: str = "neutral"
    ) -> list[TriggerSignal]:
        """Evaluate 1h OHLCV + swing_bias and return any pullback triggers.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.
            swing_bias: Context swing bias — "bullish", "bearish", or "neutral".
                        Returns [] immediately when "neutral".

        Returns:
            List of TriggerSignal (empty if no pullback/recovery conditions met).
        """
        if swing_bias == "neutral":
            return []

        if len(df) < _MIN_BARS:
            return []

        close = df["close"]

        # RSI-14 (Wilder smoothing via rolling mean)
        delta = close.diff()
        gain = delta.clip(lower=0).rolling(14).mean()
        loss = (-delta.clip(upper=0)).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))

        if rsi.isna().iloc[-1] or len(rsi) < _RECOVERY_LOOKBACK + 2:
            return []

        current_rsi = float(rsi.iloc[-1])
        # The _RECOVERY_LOOKBACK bars immediately before the current bar
        rsi_window = rsi.iloc[-_RECOVERY_LOOKBACK - 1 : -1]

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)
        signals: list[TriggerSignal] = []

        if swing_bias == "bullish":
            in_zone = _LONG_RSI_LO <= current_rsi <= _LONG_RSI_HI
            recovering = float(rsi_window.min()) < current_rsi
            if in_zone and recovering:
                depth = _LONG_RSI_HI - current_rsi          # 0 at zone top, 10 at floor
                strength = min(_BASE_STRENGTH + depth / 14.0 * 0.22, _MAX_STRENGTH)
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="pullback_1h",
                        direction="long",
                        strength=round(strength, 3),
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Pullback long: RSI {current_rsi:.1f} in"
                            f" [{_LONG_RSI_LO:.0f}–{_LONG_RSI_HI:.0f}], recovering"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "rsi": round(current_rsi, 2),
                            "rsi_3bar_min": round(float(rsi_window.min()), 2),
                        },
                    )
                )

        elif swing_bias == "bearish":
            in_zone = _SHORT_RSI_LO <= current_rsi <= _SHORT_RSI_HI
            declining = float(rsi_window.max()) > current_rsi
            if in_zone and declining:
                depth = current_rsi - _SHORT_RSI_LO         # 0 at zone floor, 10 at top
                strength = min(_BASE_STRENGTH + depth / 14.0 * 0.22, _MAX_STRENGTH)
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="pullback_1h",
                        direction="short",
                        strength=round(strength, 3),
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=(
                            f"Pullback short: RSI {current_rsi:.1f} in"
                            f" [{_SHORT_RSI_LO:.0f}–{_SHORT_RSI_HI:.0f}], declining"
                        ),
                        expires_at=expiry,
                        raw_data={
                            "rsi": round(current_rsi, 2),
                            "rsi_3bar_max": round(float(rsi_window.max()), 2),
                        },
                    )
                )

        return signals
