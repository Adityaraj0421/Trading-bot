# tests/test_context_analyzers.py
import math

import numpy as np
import pandas as pd

from context.swing import SwingAnalyzer


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
