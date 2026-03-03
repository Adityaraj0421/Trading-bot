# tests/test_context_analyzers.py
import math

import numpy as np
import pandas as pd

from context.funding import FundingAnalyzer
from context.oi_trend import OITrendAnalyzer
from context.swing import SwingAnalyzer
from context.whale_flow import WhaleFlowAnalyzer


def make_trending_up_df(n=100):
    """Steadily rising prices — should produce bullish bias."""
    closes = np.linspace(90000, 100000, n)
    df = pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    return df


def make_trending_down_df(n=100):
    """Steadily falling prices — should produce bearish bias."""
    closes = np.linspace(100000, 90000, n)
    df = pd.DataFrame(
        {
            "open": closes * 1.001,
            "high": closes * 1.005,
            "low": closes * 0.995,
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    return df


def make_sideways_df(n=84):
    """Oscillating prices — should produce neutral bias.

    n=84 chosen so the sine ends near its zero-crossing (sin(84*0.3)≈0.07),
    ensuring EMA-21 and EMA-50 partially offset each other and neither
    bullish nor bearish count reaches 2. Robust across amplitudes.
    """
    closes = [95000 + 1000 * math.sin(i * 0.3) for i in range(n)]
    df = pd.DataFrame(
        {
            "open": [c * 0.999 for c in closes],
            "high": [c * 1.002 for c in closes],
            "low": [c * 0.998 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )
    return df


class TestSwingAnalyzer:
    def test_bullish_bias_on_uptrend(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert result["swing_bias"] == "bullish"

    def test_bearish_bias_on_downtrend(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_down_df())
        assert result["swing_bias"] == "bearish"

    def test_neutral_on_sideways(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_sideways_df())
        assert result["swing_bias"] == "neutral"

    def test_returns_key_levels(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert "support" in result["key_levels"]
        assert "resistance" in result["key_levels"]
        assert "poc" in result["key_levels"]
        assert result["key_levels"]["support"] > 0

    def test_returns_allowed_directions(self):
        analyzer = SwingAnalyzer()
        result = analyzer.analyze(make_trending_up_df())
        assert "long" in result["allowed_directions"]

    def test_insufficient_data_returns_neutral(self):
        analyzer = SwingAnalyzer()
        df = make_trending_up_df(n=10)  # too few bars
        result = analyzer.analyze(df)
        assert result["swing_bias"] == "neutral"
        assert result["allowed_directions"] == []
        assert result["confidence"] == 0.0


class TestFundingAnalyzer:
    def test_positive_funding_is_long_crowded(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0005)  # 0.05%
        assert "long_crowded" in result["funding_pressure"]

    def test_negative_funding_is_short_crowded(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=-0.0006)
        assert "short_crowded" in result["funding_pressure"]

    def test_near_zero_is_neutral(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0001)
        assert result["funding_pressure"] == "neutral"

    def test_extreme_positive_funding(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=0.0012)  # > 0.10%
        assert result["funding_pressure"] == "long_crowded_extreme"

    def test_none_funding_returns_neutral(self):
        analyzer = FundingAnalyzer()
        result = analyzer.analyze(funding_rate=None)
        assert result["funding_pressure"] == "neutral"


class TestWhaleFlowAnalyzer:
    def test_positive_net_flow_is_accumulating(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=500_000.0)
        assert result["whale_flow"] == "accumulating"

    def test_negative_net_flow_is_distributing(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=-500_000.0)
        assert result["whale_flow"] == "distributing"

    def test_small_flow_is_neutral(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=1000.0)
        assert result["whale_flow"] == "neutral"

    def test_none_flow_is_neutral(self):
        analyzer = WhaleFlowAnalyzer()
        result = analyzer.analyze(net_flow=None)
        assert result["whale_flow"] == "neutral"


class TestOITrendAnalyzer:
    def test_oi_up_price_up_is_expanding_up(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=5.0, price_change_pct=3.0)
        assert result["oi_trend"] == "expanding_up"

    def test_oi_down_price_down_is_expanding_down(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=-5.0, price_change_pct=-3.0)
        assert result["oi_trend"] == "expanding_down"

    def test_oi_down_price_up_is_contracting(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=-3.0, price_change_pct=2.0)
        assert result["oi_trend"] == "contracting"

    def test_small_changes_is_neutral(self):
        analyzer = OITrendAnalyzer()
        result = analyzer.analyze(oi_change_pct=0.5, price_change_pct=0.3)
        assert result["oi_trend"] == "neutral"
