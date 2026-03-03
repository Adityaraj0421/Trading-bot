"""Multi-timeframe OHLCV fetcher.

Fetches 4h, 1h, and 15m bars from the exchange and packages them
into an immutable DataSnapshot. Each timeframe failure is isolated —
a rate-limit on 4h does not prevent 1h and 15m from being fetched.
"""

from __future__ import annotations

import logging
from typing import Any

import pandas as pd

from data_snapshot import DataSnapshot

_log = logging.getLogger(__name__)

_BARS: dict[str, int] = {
    "4h": 200,   # ~33 days for swing context
    "1h": 200,   # ~8 days for trigger analysis
    "15m": 100,  # ~25 hours for fine momentum
}

_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume"]


class MultiTimeframeFetcher:
    """Fetches 4h/1h/15m OHLCV data and returns an immutable DataSnapshot.

    Args:
        exchange: A CCXT exchange instance with fetch_ohlcv support.
    """

    def __init__(self, exchange: Any) -> None:
        self.exchange = exchange

    def fetch(self, symbol: str) -> DataSnapshot:
        """Fetch all timeframes for a symbol and return a DataSnapshot.

        Failed timeframes are returned as None (graceful degradation).

        Args:
            symbol: Trading pair, e.g. "BTC/USDT".

        Returns:
            DataSnapshot with available DataFrames and UTC timestamp.
        """
        frames: dict[str, pd.DataFrame | None] = {}
        for tf, n_bars in _BARS.items():
            try:
                raw = self.exchange.fetch_ohlcv(symbol, tf, limit=n_bars)
                df = pd.DataFrame(raw, columns=_COLUMNS)
                df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
                df = df.set_index("timestamp").sort_index()
                frames[tf] = df
            except Exception as e:
                _log.warning("Failed to fetch %s %s: %s", symbol, tf, e)
                frames[tf] = None

        return DataSnapshot(
            df_1h=frames.get("1h"),
            df_4h=frames.get("4h"),
            df_15m=frames.get("15m"),
            symbol=symbol,
        )
