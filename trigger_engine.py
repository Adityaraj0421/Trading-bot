"""TriggerEngine — orchestrates all per-candle and realtime triggers.

Maintains a TTL-aware signal buffer. Callers:
  1. Call on_1h_close() after each 1h candle close → runs MomentumTrigger,
     LiquiditySweepTrigger, PullbackTrigger (when swing_bias is non-neutral).
  2. Call on_orderflow_update() on each orderbook snapshot → runs OrderFlowTrigger.
  3. Optionally: on_liquidation_event() / on_funding_update() when
     ``USE_PHASE9_PERP=true`` — runs LiquidationTrigger / FundingExtremeTrigger.
  4. Call valid_signals() to get all non-expired signals for the Decision Layer.

The buffer is capped at max_buffer to prevent unbounded growth. Old signals
expire naturally via their expires_at field; the buffer is pruned on each write.
"""

from __future__ import annotations

import logging
import os
from collections import deque
from datetime import UTC, datetime
from typing import Any

import pandas as pd

from decision import TriggerSignal
from triggers.liquidity_sweep import LiquiditySweepTrigger
from triggers.momentum import MomentumTrigger
from triggers.orderflow import OrderFlowTrigger
from triggers.pullback import PullbackTrigger

_log = logging.getLogger(__name__)

_DEFAULT_BUFFER = 50  # Maximum cached signals per engine instance


class TriggerEngine:
    """Runs all triggers and maintains a TTL-aware signal buffer.

    Designed to be instantiated once per symbol. The Decision Layer calls
    ``valid_signals()`` to retrieve all non-expired signals accumulated since
    the last decision cycle.

    Args:
        symbol: Trading pair, e.g. "BTC/USDT".
        max_buffer: Maximum number of signals held in the buffer at once.
    """

    def __init__(self, symbol: str = "BTC/USDT", max_buffer: int = _DEFAULT_BUFFER) -> None:
        self.symbol = symbol
        self._momentum = MomentumTrigger(symbol=symbol)
        self._orderflow = OrderFlowTrigger(symbol=symbol)
        self._sweep = LiquiditySweepTrigger(symbol=symbol)
        self._pullback = PullbackTrigger(symbol=symbol)
        self._buffer: deque[TriggerSignal] = deque(maxlen=max_buffer)

        # Phase 9 perp triggers — enabled via USE_PHASE9_PERP=true
        self._use_perp = os.getenv("USE_PHASE9_PERP", "false").lower() == "true"
        if self._use_perp:
            from triggers.funding_extreme import FundingExtremeTrigger
            from triggers.liquidation import LiquidationTrigger

            self._liquidation: Any = LiquidationTrigger(symbol=symbol)
            self._funding_extreme: Any = FundingExtremeTrigger(symbol=symbol)

    def on_1h_close(self, df: pd.DataFrame, swing_bias: str = "neutral") -> list[TriggerSignal]:
        """Run per-candle triggers on 1h OHLCV data.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.
            swing_bias: Context swing bias passed from ContextEngine — "bullish",
                        "bearish", or "neutral". Forwarded to PullbackTrigger.

        Returns:
            Newly generated TriggerSignals from this candle (may be empty).
        """
        new_signals = (
            self._momentum.evaluate(df)
            + self._sweep.evaluate(df)
            + self._pullback.evaluate(df, swing_bias=swing_bias)
        )
        self._extend(new_signals)
        if new_signals:
            _log.info(
                "TriggerEngine[%s] on_1h_close: %d new signal(s)",
                self.symbol,
                len(new_signals),
            )
        return new_signals

    def on_orderflow_update(
        self,
        prices: list[float],
        cvd: list[float],
        imbalance_ratio: float,
        ratio_history: list[float],
    ) -> list[TriggerSignal]:
        """Run realtime orderflow trigger on current orderbook state.

        Args:
            prices: Recent close prices (at least 4 for CVD divergence check).
            cvd: Cumulative Volume Delta aligned to prices.
            imbalance_ratio: Current bid_volume / ask_volume ratio.
            ratio_history: Historical imbalance ratios (≥ 20 for σ calculation).

        Returns:
            Newly generated TriggerSignals from this update (may be empty).
        """
        new_signals = self._orderflow.evaluate(prices, cvd, imbalance_ratio, ratio_history)
        self._extend(new_signals)
        if new_signals:
            _log.info(
                "TriggerEngine[%s] on_orderflow_update: %d new signal(s)",
                self.symbol,
                len(new_signals),
            )
        return new_signals

    def on_liquidation_event(self, liq_data: dict[str, Any] | None) -> list[TriggerSignal]:
        """Run LiquidationTrigger on a liquidation event (perp only).

        No-op when ``USE_PHASE9_PERP`` is not set.

        Args:
            liq_data: Dict with ``liq_volume_usd`` and ``direction``, or ``None``.

        Returns:
            Newly generated TriggerSignals (may be empty).
        """
        if not self._use_perp:
            return []
        new_signals = self._liquidation.evaluate(liq_data)
        self._extend(new_signals)
        if new_signals:
            _log.info(
                "TriggerEngine[%s] liquidation: %d signal(s)",
                self.symbol,
                len(new_signals),
            )
        return new_signals

    def on_funding_update(self, funding_rate: float | None) -> list[TriggerSignal]:
        """Run FundingExtremeTrigger on the latest funding rate (perp only).

        No-op when ``USE_PHASE9_PERP`` is not set.

        Args:
            funding_rate: Current 8h funding rate as a decimal, or ``None``.

        Returns:
            Newly generated TriggerSignals (may be empty).
        """
        if not self._use_perp:
            return []
        new_signals = self._funding_extreme.evaluate(funding_rate)
        self._extend(new_signals)
        if new_signals:
            _log.info(
                "TriggerEngine[%s] funding_extreme: %d signal(s)",
                self.symbol,
                len(new_signals),
            )
        return new_signals

    def valid_signals(self) -> list[TriggerSignal]:
        """Return all non-expired signals from the buffer.

        The Decision Layer passes this list to ``evaluate()`` in decision.py.
        Expired signals are filtered out but not removed from the buffer here;
        they will be evicted on the next ``_extend()`` call.

        Returns:
            List of unexpired TriggerSignals, newest last.
        """
        now = datetime.now(UTC)
        return [s for s in self._buffer if s.expires_at > now]

    def clear(self) -> None:
        """Discard all buffered signals (e.g. after a risk supervisor reset)."""
        self._buffer.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extend(self, signals: list[TriggerSignal]) -> None:
        """Add new signals to the buffer, pruning expired entries first."""
        now = datetime.now(UTC)
        # Replace buffer with only unexpired signals + new ones.
        # deque(maxlen=) handles overflow automatically.
        unexpired = deque(
            (s for s in self._buffer if s.expires_at > now),
            maxlen=self._buffer.maxlen,
        )
        unexpired.extend(signals)
        self._buffer = unexpired
