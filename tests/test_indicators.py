"""
Unit tests for indicators.py — Indicators.add_all(), caching, FEATURE_COLUMNS.

Validates that all technical indicators are computed correctly, values are
in expected ranges, and caching works as expected.
"""

import pytest
import pandas as pd
import numpy as np
from demo_data import generate_ohlcv
from indicators import Indicators, FEATURE_COLUMNS


@pytest.fixture()
def sample_ohlcv():
    return generate_ohlcv(periods=200, seed=42)


@pytest.fixture(autouse=True)
def _clear_cache():
    """Ensure indicator cache is clean before each test."""
    Indicators.invalidate_cache()
    yield
    Indicators.invalidate_cache()


# ---------------------------------------------------------------------------
# add_all — output shape and columns
# ---------------------------------------------------------------------------

class TestIndicatorsAddAll:
    def test_returns_dataframe(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert isinstance(result, pd.DataFrame)

    def test_all_feature_columns_present(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        for col in FEATURE_COLUMNS:
            assert col in result.columns, f"Missing feature column: {col}"

    def test_sma_columns_exist(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        for col in ["sma_10", "sma_20", "sma_50"]:
            assert col in result.columns

    def test_ema_columns_exist(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        for col in ["ema_12", "ema_26"]:
            assert col in result.columns

    def test_macd_columns_exist(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        for col in ["macd", "macd_signal", "macd_hist"]:
            assert col in result.columns

    def test_rsi_range(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["rsi"] >= 0).all()
        assert (result["rsi"] <= 100).all()

    def test_stochastic_range(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["stoch_k"] >= 0).all()
        assert (result["stoch_k"] <= 100).all()
        assert (result["stoch_d"] >= 0).all()
        assert (result["stoch_d"] <= 100).all()

    def test_bb_upper_above_lower(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["bb_upper"] > result["bb_lower"]).all()

    def test_atr_positive(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["atr"] > 0).all()

    def test_atr_pct_reasonable(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["atr_pct"] > 0).all()
        assert (result["atr_pct"] < 0.5).all()

    def test_volume_ratio_positive(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert (result["volume_ratio"] > 0).all()

    def test_returns_columns_exist(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        for col in ["returns_1", "returns_5", "returns_10"]:
            assert col in result.columns

    def test_log_returns_exist(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert "log_returns" in result.columns
        assert "rolling_vol_10" in result.columns

    def test_future_return_exists(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert "future_return" in result.columns

    def test_no_nan_values(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert not result.isna().any().any()

    def test_output_shorter_than_input(self, sample_ohlcv):
        result = Indicators.add_all(sample_ohlcv)
        assert len(result) < len(sample_ohlcv)


# ---------------------------------------------------------------------------
# Caching
# ---------------------------------------------------------------------------

class TestIndicatorsCache:
    def test_cache_hit_same_data(self, sample_ohlcv):
        r1 = Indicators.add_all(sample_ohlcv)
        r2 = Indicators.add_all(sample_ohlcv)
        assert r1 is r2  # Same object reference

    def test_cache_miss_different_data(self, sample_ohlcv):
        r1 = Indicators.add_all(sample_ohlcv)
        df2 = sample_ohlcv.copy()
        df2.iloc[-1, df2.columns.get_loc("close")] += 100  # Modify last close
        r2 = Indicators.add_all(df2)
        assert r1 is not r2

    def test_invalidate_cache(self, sample_ohlcv):
        r1 = Indicators.add_all(sample_ohlcv)
        Indicators.invalidate_cache()
        r2 = Indicators.add_all(sample_ohlcv)
        assert r1 is not r2  # Recomputed


# ---------------------------------------------------------------------------
# FEATURE_COLUMNS
# ---------------------------------------------------------------------------

class TestFeatureColumns:
    def test_get_feature_columns_returns_list(self):
        cols = Indicators.get_feature_columns()
        assert isinstance(cols, list)

    def test_feature_columns_count(self):
        # v9.0: expanded from 15 to 22 features
        # (added obv_divergence, close_to_vwap, ema_cross, adx, plus_di, minus_di, rolling_vol_10)
        cols = Indicators.get_feature_columns()
        assert len(cols) == 22
