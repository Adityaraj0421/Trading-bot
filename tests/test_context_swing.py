# tests/test_context_swing.py
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from context.swing import SwingAnalyzer


def make_df_with_dates(n_days: int = 10, bars_per_day: int = 6) -> pd.DataFrame:
    """4h OHLCV with a real DatetimeIndex spanning n_days.

    bars_per_day=6 represents 4h bars (6 × 4h = 24h).
    n_days=10 → 60 bars, which exceeds SwingAnalyzer._MIN_BARS=50.
    """
    n = n_days * bars_per_day
    closes = np.linspace(90000.0, 100000.0, n)
    start = datetime(2026, 3, 1, 0, 0, tzinfo=UTC)
    index = [start + timedelta(hours=4 * i) for i in range(n)]
    return pd.DataFrame(
        {
            "open":   closes * 0.999,
            "high":   closes * 1.01,
            "low":    closes * 0.99,
            "close":  closes,
            "volume": [1000.0] * n,
        },
        index=pd.DatetimeIndex(index),
    )


class TestSwingAnalyzerPdh:
    def test_pdh_and_pdl_present_with_datetime_index(self):
        # 10 days × 6 bars = 60 bars ≥ _MIN_BARS=50; 2+ distinct dates → pdh/pdl present
        df = make_df_with_dates(n_days=10, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert "pdh" in result["key_levels"]
        assert "pdl" in result["key_levels"]

    def test_pdh_greater_than_pdl(self):
        df = make_df_with_dates(n_days=10, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert result["key_levels"]["pdh"] > result["key_levels"]["pdl"]

    def test_pdh_pdl_absent_without_datetime_index(self):
        # Integer-indexed df (existing tests use this) → no pdh/pdl
        n = 60
        closes = np.linspace(90000.0, 100000.0, n)
        df = pd.DataFrame({
            "open":  closes * 0.999,
            "high":  closes * 1.005,
            "low":   closes * 0.995,
            "close": closes,
            "volume": [1000.0] * n,
        })
        result = SwingAnalyzer().analyze(df)
        assert "pdh" not in result["key_levels"]
        assert "pdl" not in result["key_levels"]

    def test_pdh_pdl_absent_with_only_one_day(self):
        # 1 day = 6 bars < _MIN_BARS=50; early return → no pdh/pdl
        df = make_df_with_dates(n_days=1, bars_per_day=6)
        result = SwingAnalyzer().analyze(df)
        assert "pdh" not in result["key_levels"]
