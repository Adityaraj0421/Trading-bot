"""
Unit tests for indicators.py — Indicators.add_all(), caching, FEATURE_COLUMNS.

Validates that all technical indicators are computed correctly, values are
in expected ranges, and caching works as expected.
"""

import pandas as pd
import pytest

from demo_data import generate_ohlcv
from indicators import FEATURE_COLUMNS, Indicators


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
        # v9.1: expanded to 24 features (added williams_r, cci)
        cols = Indicators.get_feature_columns()
        assert len(cols) == 24

    def test_williams_r_in_feature_columns(self):
        assert "williams_r" in Indicators.get_feature_columns()

    def test_cci_in_feature_columns(self):
        assert "cci" in Indicators.get_feature_columns()

    def test_williams_r_range(self, sample_ohlcv):
        """Williams %R must always lie in [-100, 0]."""
        result = Indicators.add_all(sample_ohlcv)
        assert "williams_r" in result.columns
        assert (result["williams_r"] >= -100).all()
        assert (result["williams_r"] <= 0).all()

    def test_cci_exists(self, sample_ohlcv):
        """CCI column should be present after add_all()."""
        result = Indicators.add_all(sample_ohlcv)
        assert "cci" in result.columns

    def test_ichimoku_columns_exist(self, sample_ohlcv):
        """All 7 Ichimoku columns should be present after add_all()."""
        result = Indicators.add_all(sample_ohlcv)
        for col in [
            "ichimoku_tenkan",
            "ichimoku_kijun",
            "ichimoku_span_a",
            "ichimoku_span_b",
            "ichimoku_above_cloud",
            "ichimoku_below_cloud",
            "ichimoku_tk_cross",
        ]:
            assert col in result.columns, f"Missing Ichimoku column: {col}"

    def test_ichimoku_not_in_feature_columns(self):
        """Ichimoku columns are directional flags — not ML numeric features."""
        cols = Indicators.get_feature_columns()
        for col in ["ichimoku_tenkan", "ichimoku_kijun", "ichimoku_above_cloud"]:
            assert col not in cols, f"{col} should not be in FEATURE_COLUMNS"
