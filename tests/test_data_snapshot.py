# tests/test_data_snapshot.py
from datetime import datetime

import pandas as pd
import pytest

from data_snapshot import DataSnapshot


def make_df(n: int = 10) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "open": [100.0] * n,
            "high": [101.0] * n,
            "low": [99.0] * n,
            "close": [100.5] * n,
            "volume": [1000.0] * n,
        },
        index=idx,
    )


class TestDataSnapshot:
    def test_snapshot_stores_dataframes(self):
        df = make_df(10)
        snap = DataSnapshot(df_1h=df, df_4h=None, df_15m=None)
        assert len(snap.df_1h) == 10

    def test_snapshot_is_timestamped(self):
        snap = DataSnapshot(df_1h=make_df(), df_4h=None, df_15m=None)
        assert isinstance(snap.captured_at, datetime)
        assert snap.captured_at.tzinfo is not None  # must be tz-aware

    def test_snapshot_df_is_readonly(self):
        snap = DataSnapshot(df_1h=make_df(), df_4h=None, df_15m=None)
        with pytest.raises(ValueError):
            snap.df_1h.iloc[0, 0] = 999.0

    def test_snapshot_has_symbol(self):
        snap = DataSnapshot(df_1h=make_df(), df_4h=None, df_15m=None, symbol="BTC/USDT")
        assert snap.symbol == "BTC/USDT"

    def test_snapshot_accepts_none_timeframes(self):
        snap = DataSnapshot(df_1h=None, df_4h=None, df_15m=None)
        assert snap.df_1h is None
        assert snap.df_4h is None
        assert snap.df_15m is None

    def test_snapshot_4h_is_readonly(self):
        snap = DataSnapshot(df_1h=None, df_4h=make_df(), df_15m=None)
        with pytest.raises(ValueError):
            snap.df_4h.iloc[0, 0] = 999.0

    def test_default_symbol(self):
        snap = DataSnapshot(df_1h=make_df(), df_4h=None, df_15m=None)
        assert snap.symbol == "BTC/USDT"
