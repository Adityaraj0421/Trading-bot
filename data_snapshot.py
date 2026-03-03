"""Data snapshot — immutable timestamped OHLCV data container.

All market data enters the system through DataSnapshot. Downstream
components read from snapshots only — they never fetch raw data directly.
This ensures consistent state within a processing cycle and makes
debugging deterministic (the snapshot is the ground truth for that cycle).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime

import pandas as pd


def _freeze(df: pd.DataFrame) -> None:
    """Set all underlying NumPy arrays in *df* to read-only in place."""
    for arr in df._mgr.arrays:  # noqa: SLF001
        arr.flags.writeable = False


@dataclass
class DataSnapshot:
    """Immutable multi-timeframe OHLCV snapshot.

    DataFrames are set to read-only at construction time to prevent
    accidental mutation downstream. None values indicate that timeframe
    data was not available for this snapshot.

    Attributes:
        df_1h: 1-hour OHLCV DataFrame (primary timeframe for triggers).
        df_4h: 4-hour OHLCV DataFrame (used by SwingAnalyzer).
        df_15m: 15-minute OHLCV DataFrame (used for fine-grained momentum).
        symbol: Trading pair, e.g. "BTC/USDT".
        captured_at: UTC datetime when this snapshot was taken.
    """

    df_1h: pd.DataFrame | None
    df_4h: pd.DataFrame | None
    df_15m: pd.DataFrame | None
    symbol: str = "BTC/USDT"
    captured_at: datetime = field(default_factory=lambda: datetime.now(UTC))

    def __post_init__(self) -> None:
        for attr in ("df_1h", "df_4h", "df_15m"):
            df = getattr(self, attr)
            if df is not None:
                _freeze(df)
