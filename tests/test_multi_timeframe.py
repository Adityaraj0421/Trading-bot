"""
Unit tests for multi_timeframe.py — MultiTimeframeConfirmer resample, bias, signal confirmation.

Tests resample aggregation, HTF bias computation, cache behaviour,
and the pure confirm_signal adjustment logic.
"""

import pandas as pd
import pytest

from demo_data import generate_ohlcv
from indicators import Indicators
from multi_timeframe import MultiTimeframeConfirmer


@pytest.fixture()
def confirmer():
    return MultiTimeframeConfirmer()


@pytest.fixture()
def hourly_df():
    return generate_ohlcv(periods=200, seed=42)


@pytest.fixture(autouse=True)
def _clear_indicator_cache():
    Indicators.invalidate_cache()
    yield
    Indicators.invalidate_cache()


# ---------------------------------------------------------------------------
# resample_to_higher_tf
# ---------------------------------------------------------------------------


class TestResampleToHigherTf:
    def test_reduces_row_count(self, confirmer, hourly_df):
        htf = confirmer.resample_to_higher_tf(hourly_df, target_tf="4h")
        assert len(htf) < len(hourly_df)
        # ~200 hourly bars → ~50 four-hourly bars
        assert len(htf) == pytest.approx(50, abs=5)

    def test_open_uses_first(self, confirmer):
        idx = pd.date_range("2024-01-01", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10, 20, 30, 40, 50, 60, 70, 80],
                "high": [15, 25, 35, 45, 55, 65, 75, 85],
                "low": [5, 15, 25, 35, 45, 55, 65, 75],
                "close": [12, 22, 32, 42, 52, 62, 72, 82],
                "volume": [100] * 8,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        htf = confirmer.resample_to_higher_tf(df, target_tf="4h")
        assert htf["open"].iloc[0] == 10

    def test_high_uses_max(self, confirmer):
        idx = pd.date_range("2024-01-01", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10] * 8,
                "high": [15, 25, 35, 45, 55, 65, 75, 85],
                "low": [5] * 8,
                "close": [12] * 8,
                "volume": [100] * 8,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        htf = confirmer.resample_to_higher_tf(df, target_tf="4h")
        assert htf["high"].iloc[0] == 45

    def test_low_uses_min(self, confirmer):
        idx = pd.date_range("2024-01-01", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10] * 8,
                "high": [50] * 8,
                "low": [5, 4, 3, 2, 6, 7, 8, 9],
                "close": [12] * 8,
                "volume": [100] * 8,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        htf = confirmer.resample_to_higher_tf(df, target_tf="4h")
        assert htf["low"].iloc[0] == 2

    def test_close_uses_last(self, confirmer):
        idx = pd.date_range("2024-01-01", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10] * 8,
                "high": [50] * 8,
                "low": [5] * 8,
                "close": [12, 22, 32, 42, 52, 62, 72, 82],
                "volume": [100] * 8,
            },
            index=idx,
        )
        df.index.name = "timestamp"
        htf = confirmer.resample_to_higher_tf(df, target_tf="4h")
        assert htf["close"].iloc[0] == 42

    def test_volume_uses_sum(self, confirmer):
        idx = pd.date_range("2024-01-01", periods=8, freq="1h")
        df = pd.DataFrame(
            {
                "open": [10] * 8,
                "high": [50] * 8,
                "low": [5] * 8,
                "close": [12] * 8,
                "volume": [100, 200, 300, 400, 500, 600, 700, 800],
            },
            index=idx,
        )
        df.index.name = "timestamp"
        htf = confirmer.resample_to_higher_tf(df, target_tf="4h")
        assert htf["volume"].iloc[0] == 1000

    def test_target_tf_mapping(self, confirmer, hourly_df):
        # "1d" should use "1D" rule, producing far fewer rows
        htf_1d = confirmer.resample_to_higher_tf(hourly_df, target_tf="1d")
        htf_4h = confirmer.resample_to_higher_tf(hourly_df, target_tf="4h")
        assert len(htf_1d) < len(htf_4h)
        # Unknown tf defaults to "4h"
        htf_unknown = confirmer.resample_to_higher_tf(hourly_df, target_tf="weird")
        assert len(htf_unknown) == len(htf_4h)


# ---------------------------------------------------------------------------
# get_htf_bias
# ---------------------------------------------------------------------------


class TestGetHtfBias:
    def test_returns_required_keys(self, confirmer, hourly_df):
        result = confirmer.get_htf_bias(hourly_df)
        assert "bias" in result
        assert "strength" in result
        assert "details" in result

    def test_insufficient_htf_data_neutral(self, confirmer):
        df = generate_ohlcv(periods=40, seed=42)
        result = confirmer.get_htf_bias(df)
        assert result["bias"] == "neutral"
        assert result["strength"] == 0.0

    def test_bias_is_valid_string(self, confirmer, hourly_df):
        result = confirmer.get_htf_bias(hourly_df)
        assert result["bias"] in ("bullish", "bearish", "neutral")

    def test_strength_bounded(self, confirmer, hourly_df):
        result = confirmer.get_htf_bias(hourly_df)
        assert 0.0 <= result["strength"] <= 1.0

    def test_cache_hit_same_data(self, confirmer, hourly_df):
        r1 = confirmer.get_htf_bias(hourly_df)
        r2 = confirmer.get_htf_bias(hourly_df)
        assert r1 is r2  # Same cached dict

    def test_cache_miss_different_data(self, confirmer, hourly_df):
        r1 = confirmer.get_htf_bias(hourly_df)
        modified = hourly_df.copy()
        modified.iloc[-1, modified.columns.get_loc("close")] += 1000.0
        r2 = confirmer.get_htf_bias(modified)
        assert r1 is not r2


# ---------------------------------------------------------------------------
# confirm_signal — pure logic
# ---------------------------------------------------------------------------


class TestConfirmSignal:
    def test_hold_unchanged(self, confirmer):
        bias = {"bias": "bullish", "strength": 1.0}
        signal, conf = confirmer.confirm_signal("HOLD", 0.7, bias)
        assert signal == "HOLD"
        assert conf == 0.7

    def test_buy_bullish_boosts(self, confirmer):
        bias = {"bias": "bullish", "strength": 0.8}
        signal, conf = confirmer.confirm_signal("BUY", 0.7, bias)
        assert signal == "BUY"
        expected = 0.7 + 0.15 * 0.8  # 0.82
        assert conf == pytest.approx(expected, abs=1e-10)

    def test_buy_bearish_reduces(self, confirmer):
        bias = {"bias": "bearish", "strength": 0.6}
        signal, conf = confirmer.confirm_signal("BUY", 0.8, bias)
        # v2.0: reduction factor changed from 0.4 to 0.25
        adjusted = 0.8 * (1 - 0.25 * 0.6)  # 0.68
        assert adjusted >= 0.3  # Should NOT flip to HOLD (threshold now 0.3)
        assert signal == "BUY"
        assert conf == pytest.approx(adjusted, abs=1e-10)

    def test_buy_bearish_flips_to_hold(self, confirmer):
        bias = {"bias": "bearish", "strength": 1.0}
        # v2.0: adjusted = 0.35 * (1 - 0.25 * 1.0) = 0.2625 < 0.3 → HOLD
        signal, conf = confirmer.confirm_signal("BUY", 0.35, bias)
        assert signal == "HOLD"
        assert conf < 0.3

    def test_buy_neutral_unchanged(self, confirmer):
        bias = {"bias": "neutral", "strength": 0.3}
        signal, conf = confirmer.confirm_signal("BUY", 0.7, bias)
        assert signal == "BUY"
        assert conf == 0.7

    def test_sell_bearish_boosts(self, confirmer):
        bias = {"bias": "bearish", "strength": 0.8}
        signal, conf = confirmer.confirm_signal("SELL", 0.7, bias)
        assert signal == "SELL"
        expected = 0.7 + 0.15 * 0.8  # 0.82
        assert conf == pytest.approx(expected, abs=1e-10)

    def test_sell_bullish_reduces(self, confirmer):
        bias = {"bias": "bullish", "strength": 0.6}
        signal, conf = confirmer.confirm_signal("SELL", 0.8, bias)
        # v2.0: reduction factor changed from 0.4 to 0.25
        adjusted = 0.8 * (1 - 0.25 * 0.6)  # 0.68
        assert signal == "SELL"
        assert conf == pytest.approx(adjusted, abs=1e-10)

    def test_sell_bullish_flips_to_hold(self, confirmer):
        bias = {"bias": "bullish", "strength": 1.0}
        # v2.0: adjusted = 0.35 * (1 - 0.25 * 1.0) = 0.2625 < 0.3 → HOLD
        signal, conf = confirmer.confirm_signal("SELL", 0.35, bias)
        assert signal == "HOLD"
        assert conf < 0.3

    def test_confidence_capped_at_095(self, confirmer):
        bias = {"bias": "bullish", "strength": 1.0}
        # 0.9 + 0.15 * 1.0 = 1.05, should be capped at 0.95
        signal, conf = confirmer.confirm_signal("BUY", 0.9, bias)
        assert conf == 0.95
