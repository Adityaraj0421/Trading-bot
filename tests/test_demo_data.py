"""
Unit tests for demo_data.py — generate_ohlcv() deterministic data generator.

Validates shape, columns, OHLCV invariants, determinism, and timestamp spacing.
"""

import pytest
import numpy as np
import pandas as pd
from demo_data import generate_ohlcv


class TestGenerateOHLCV:
    def test_returns_dataframe(self):
        df = generate_ohlcv()
        assert isinstance(df, pd.DataFrame)

    def test_default_shape(self):
        df = generate_ohlcv()
        assert len(df) == 200

    def test_custom_periods(self):
        df = generate_ohlcv(periods=50)
        assert len(df) == 50

    def test_columns(self):
        df = generate_ohlcv()
        assert list(df.columns) == ["open", "high", "low", "close", "volume"]

    def test_index_is_datetime(self):
        df = generate_ohlcv()
        assert isinstance(df.index, pd.DatetimeIndex)
        assert df.index.name == "timestamp"

    def test_deterministic_with_seed(self):
        df1 = generate_ohlcv(seed=42)
        df2 = generate_ohlcv(seed=42)
        pd.testing.assert_frame_equal(df1, df2)

    def test_different_seeds_differ(self):
        df1 = generate_ohlcv(seed=42)
        df2 = generate_ohlcv(seed=99)
        assert not df1["close"].equals(df2["close"])

    def test_high_gte_close(self):
        df = generate_ohlcv()
        assert (df["high"] >= df["close"]).all()

    def test_high_gte_open(self):
        df = generate_ohlcv()
        assert (df["high"] >= df["open"]).all()

    def test_low_lte_close(self):
        df = generate_ohlcv()
        assert (df["low"] <= df["close"]).all()

    def test_low_lte_open(self):
        df = generate_ohlcv()
        assert (df["low"] <= df["open"]).all()

    def test_volume_positive(self):
        df = generate_ohlcv()
        assert (df["volume"] > 0).all()

    def test_prices_positive(self):
        df = generate_ohlcv()
        for col in ["open", "high", "low", "close"]:
            assert (df[col] > 0).all()

    def test_start_price_respected(self):
        df = generate_ohlcv(start_price=50000.0)
        # First close should be within 10% of start_price
        assert abs(df["close"].iloc[0] / 50000.0 - 1) < 0.10

    def test_timestamps_monotonic(self):
        df = generate_ohlcv()
        assert df.index.is_monotonic_increasing

    def test_timestamps_spacing(self):
        df = generate_ohlcv(timeframe_minutes=60)
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=60)).all()

    def test_custom_timeframe(self):
        df = generate_ohlcv(timeframe_minutes=15)
        diffs = df.index.to_series().diff().dropna()
        assert (diffs == pd.Timedelta(minutes=15)).all()

    def test_no_nan_values(self):
        df = generate_ohlcv()
        assert not df.isna().any().any()
