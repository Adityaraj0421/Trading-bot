"""OrderFlowTrigger — CVD divergence + bid/ask imbalance spike.

Fires when at least one of these conditions is true:
  - CVD divergence: price makes new high/low but CVD does not confirm
    (price running on declining participation = exhaustion signal)
  - Bid/ask imbalance ratio spike > 2σ above/below rolling mean
    (sudden dominance of one side = directional pressure signal)

Each condition emits a separate TriggerSignal with urgency="normal".
"""

from __future__ import annotations

import logging
import uuid
from datetime import UTC, datetime, timedelta

import numpy as np

from decision import TriggerSignal

_log = logging.getLogger(__name__)

_SIGNAL_TTL_MINUTES = 15   # Shorter than momentum — realtime data decays faster
_CVD_LOOKBACK = 3          # Bars to look back for divergence check
_SIGMA_MULT = 2.0          # Sigma multiplier for imbalance spike
_MIN_RATIO_HISTORY = 20    # Minimum samples needed for σ calculation
_CVD_STRENGTH = 0.55       # Fixed strength for CVD divergence signal
_IMBALANCE_STRENGTH_BASE = 0.40
_IMBALANCE_STRENGTH_MAX = 0.85


class OrderFlowTrigger:
    """Generates order-flow-based TriggerSignals from realtime market data.

    Intended to run on each orderbook update (or at least per 1h candle).
    Produces signals with ``urgency="normal"`` — they feed the spot executor
    via the Decision Layer, not the perp executor.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT". Used to set symbol_scope.
    """

    def __init__(self, symbol: str = "BTC/USDT") -> None:
        self.symbol = symbol
        self._symbol_scope = symbol.split("/")[0]  # "BTC/USDT" → "BTC"

    def evaluate(
        self,
        prices: list[float],
        cvd: list[float],
        imbalance_ratio: float,
        ratio_history: list[float],
    ) -> list[TriggerSignal]:
        """Evaluate orderflow data and return any triggered signals.

        Args:
            prices: Recent close prices. Needs at least _CVD_LOOKBACK + 1 values.
            cvd: Cumulative Volume Delta aligned to prices (same length).
            imbalance_ratio: Current bid_volume / ask_volume ratio. > 1 = bid heavy.
            ratio_history: Historical imbalance ratios for σ calculation.
                Must contain at least _MIN_RATIO_HISTORY values for imbalance
                spike detection; shorter history skips that check.

        Returns:
            List of TriggerSignal (empty if no orderflow conditions met).
        """
        signals: list[TriggerSignal] = []
        now = datetime.now(UTC)
        expiry = now + timedelta(minutes=_SIGNAL_TTL_MINUTES)

        # --- CVD divergence check ---
        if len(prices) >= _CVD_LOOKBACK + 1 and len(cvd) >= _CVD_LOOKBACK + 1:
            current_price = prices[-1]
            lookback_price = prices[-1 - _CVD_LOOKBACK]
            current_cvd = cvd[-1]
            lookback_cvd = cvd[-1 - _CVD_LOOKBACK]

            price_new_high = current_price > lookback_price
            price_new_low = current_price < lookback_price
            cvd_confirming_high = current_cvd > lookback_cvd
            cvd_confirming_low = current_cvd < lookback_cvd

            if price_new_high and not cvd_confirming_high:
                # Price rose but CVD didn't confirm → exhaustion → short
                reason = (
                    f"CVD divergence: price +{current_price - lookback_price:.0f}"
                    f" but CVD {current_cvd - lookback_cvd:+.0f} (non-confirming)"
                )
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="orderflow",
                        direction="short",
                        strength=_CVD_STRENGTH,
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=reason,
                        expires_at=expiry,
                        raw_data={
                            "type": "cvd_divergence",
                            "price_change": round(current_price - lookback_price, 2),
                            "cvd_change": round(current_cvd - lookback_cvd, 4),
                        },
                    )
                )

            if price_new_low and not cvd_confirming_low:
                # Price fell but CVD didn't confirm → exhaustion → long
                reason = (
                    f"CVD divergence: price {current_price - lookback_price:+.0f}"
                    f" but CVD {current_cvd - lookback_cvd:+.0f} (non-confirming)"
                )
                signals.append(
                    TriggerSignal(
                        trigger_id=str(uuid.uuid4()),
                        source="orderflow",
                        direction="long",
                        strength=_CVD_STRENGTH,
                        urgency="normal",
                        symbol_scope=self._symbol_scope,
                        reason=reason,
                        expires_at=expiry,
                        raw_data={
                            "type": "cvd_divergence",
                            "price_change": round(current_price - lookback_price, 2),
                            "cvd_change": round(current_cvd - lookback_cvd, 4),
                        },
                    )
                )

        # --- Bid/ask imbalance spike check ---
        if len(ratio_history) >= _MIN_RATIO_HISTORY:
            arr = np.array(ratio_history, dtype=float)
            mean = float(np.mean(arr))
            std = float(np.std(arr))

            if std > 0:
                deviations = abs(imbalance_ratio - mean) / std

                if imbalance_ratio > mean + _SIGMA_MULT * std:
                    # Bid-heavy spike → buyers dominating → long signal
                    strength = min(
                        _IMBALANCE_STRENGTH_BASE + deviations * 0.1,
                        _IMBALANCE_STRENGTH_MAX,
                    )
                    reason = (
                        f"Imbalance spike: ratio {imbalance_ratio:.2f}"
                        f" = +{deviations:.1f}σ above mean ({mean:.2f})"
                    )
                    signals.append(
                        TriggerSignal(
                            trigger_id=str(uuid.uuid4()),
                            source="orderflow",
                            direction="long",
                            strength=strength,
                            urgency="normal",
                            symbol_scope=self._symbol_scope,
                            reason=reason,
                            expires_at=expiry,
                            raw_data={
                                "type": "imbalance_spike",
                                "ratio": round(imbalance_ratio, 4),
                                "mean": round(mean, 4),
                                "std": round(std, 4),
                                "deviations": round(deviations, 2),
                            },
                        )
                    )

                elif imbalance_ratio < mean - _SIGMA_MULT * std:
                    # Ask-heavy spike → sellers dominating → short signal
                    strength = min(
                        _IMBALANCE_STRENGTH_BASE + deviations * 0.1,
                        _IMBALANCE_STRENGTH_MAX,
                    )
                    reason = (
                        f"Imbalance spike: ratio {imbalance_ratio:.2f}"
                        f" = -{deviations:.1f}σ below mean ({mean:.2f})"
                    )
                    signals.append(
                        TriggerSignal(
                            trigger_id=str(uuid.uuid4()),
                            source="orderflow",
                            direction="short",
                            strength=strength,
                            urgency="normal",
                            symbol_scope=self._symbol_scope,
                            reason=reason,
                            expires_at=expiry,
                            raw_data={
                                "type": "imbalance_spike",
                                "ratio": round(imbalance_ratio, 4),
                                "mean": round(mean, 4),
                                "std": round(std, 4),
                                "deviations": round(deviations, 2),
                            },
                        )
                    )

        return signals
