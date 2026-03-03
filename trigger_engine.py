"""TriggerEngine — orchestrates all per-candle and realtime triggers.

Maintains a TTL-aware signal buffer. Callers:
  1. Call on_1h_close() after each 1h candle close → runs MomentumTrigger.
  2. Call on_orderflow_update() on each orderbook snapshot → runs OrderFlowTrigger.
  3. Call valid_signals() to get all non-expired signals for the Decision Layer.

The buffer is capped at max_buffer to prevent unbounded growth. Old signals
expire naturally via their expires_at field; the buffer is pruned on each write.
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime

import pandas as pd

from decision import TriggerSignal
from triggers.momentum import MomentumTrigger
from triggers.orderflow import OrderFlowTrigger

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
        self._buffer: deque[TriggerSignal] = deque(maxlen=max_buffer)

    def on_1h_close(self, df: pd.DataFrame) -> list[TriggerSignal]:
        """Run per-candle triggers on 1h OHLCV data.

        Args:
            df: 1h OHLCV DataFrame with columns: open, high, low, close, volume.

        Returns:
            Newly generated TriggerSignals from this candle (may be empty).
        """
        new_signals = self._momentum.evaluate(df)
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
