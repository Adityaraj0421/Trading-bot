"""LiquiditySweepTrigger — equal highs/lows sweep detection.

Fires when price wicks through an equal-highs or equal-lows cluster
(>=2 bars within 0.3%) and closes back inside — indicating a stop-hunt
reversal rather than a genuine breakout.
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_LOOKBACK = 20
_CLUSTER_TOL = 0.003   # 0.3% tolerance for "equal" highs/lows
_MIN_BARS = _LOOKBACK + 1
_SIGNAL_TTL_MINUTES = 75
_SWEEP_STRENGTH = 0.65


class LiquiditySweepTrigger:
    """Detects equal-highs/lows sweep patterns on 1h OHLCV data.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Sets symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]

    def evaluate(self, df: pd.DataFrame) -> list[TriggerSignal]:
        """Scan the last 20 bars (excluding current) for sweep patterns.

        Args:
            df: 1h OHLCV DataFrame. Needs >=21 rows.

        Returns:
            List of TriggerSignal (empty if no sweep detected).
        """
        if len(df) < _MIN_BARS:
            return []

        # Exclude the current bar so the cluster is built from prior bars only
        window = df.iloc[:-1].tail(_LOOKBACK)
        current = df.iloc[-1]

        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)
        signals: list[TriggerSignal] = []

        # Equal highs -> bearish sweep zone
        equal_highs_zone = self._find_extreme_cluster(window["high"].values, find_max=True)
        if equal_highs_zone is not None and current["high"] > equal_highs_zone and current["close"] < equal_highs_zone:
            signals.append(
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="liquidity_sweep",
                    direction="short",
                    strength=_SWEEP_STRENGTH,
                    urgency="normal",
                    symbol_scope=self._symbol_scope,
                    reason=(
                        f"Bearish sweep: wick {current['high']:.0f}"
                        f" > zone {equal_highs_zone:.0f},"
                        f" close {current['close']:.0f}"
                    ),
                    expires_at=expiry,
                    raw_data={
                        "type": "equal_highs_sweep",
                        "zone": round(float(equal_highs_zone), 2),
                    },
                )
            )

        # Equal lows -> bullish sweep zone
        equal_lows_zone = self._find_extreme_cluster(window["low"].values, find_max=False)
        if equal_lows_zone is not None and current["low"] < equal_lows_zone and current["close"] > equal_lows_zone:
            signals.append(
                TriggerSignal(
                    trigger_id=str(uuid.uuid4()),
                    source="liquidity_sweep",
                    direction="long",
                    strength=_SWEEP_STRENGTH,
                    urgency="normal",
                    symbol_scope=self._symbol_scope,
                    reason=(
                        f"Bullish sweep: wick {current['low']:.0f}"
                        f" < zone {equal_lows_zone:.0f},"
                        f" close {current['close']:.0f}"
                    ),
                    expires_at=expiry,
                    raw_data={
                        "type": "equal_lows_sweep",
                        "zone": round(float(equal_lows_zone), 2),
                    },
                )
            )

        return signals

    def _find_extreme_cluster(
        self, values: np.ndarray, find_max: bool
    ) -> float | None:
        """Return the extreme value if >=2 bars cluster within _CLUSTER_TOL%.

        For equal highs (find_max=True): clusters near the maximum.
        For equal lows (find_max=False): clusters near the minimum.

        Args:
            values: Array of high or low prices.
            find_max: True to find equal highs, False for equal lows.

        Returns:
            The extreme zone price, or None if no cluster found.
        """
        extreme = float(values.max() if find_max else values.min())
        if extreme == 0.0:
            return None
        cluster_count = int(np.sum(np.abs(values - extreme) / extreme <= _CLUSTER_TOL))
        return extreme if cluster_count >= 2 else None
