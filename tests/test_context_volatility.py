# tests/test_context_volatility.py
import numpy as np
import pandas as pd

from context.volatility import VolatilityAnalyzer


def make_df(n: int = 60, atr_multiplier: float = 1.0) -> pd.DataFrame:
    """Build 1h OHLCV DataFrame with controllable ATR size."""
    closes = np.linspace(90000.0, 91000.0, n)
    spread = 500.0 * atr_multiplier
    return pd.DataFrame({
        "open":   closes * 0.999,
        "high":   closes + spread,
        "low":    closes - spread,
        "close":  closes,
        "volume": [1000.0] * n,
    })


class TestVolatilityAnalyzer:
    def test_returns_dict_with_key(self):
        result = VolatilityAnalyzer().analyze(make_df())
        assert "volatility_regime" in result

    def test_normal_regime_baseline(self):
        # Stable ATR throughout → ratio ≈ 1.0 → normal
        result = VolatilityAnalyzer().analyze(make_df(n=60))
        assert result["volatility_regime"] == "normal"

    def test_extreme_regime_when_atr_spikes(self):
        # First 44 bars low ATR, last 16 bars 10× ATR → ratio >> 2.5 → extreme
        n_low, n_high = 44, 16
        closes = np.linspace(90000.0, 91000.0, n_low + n_high)
        spread_low = 100.0
        spread_high = 10000.0
        spreads = [spread_low] * n_low + [spread_high] * n_high
        df = pd.DataFrame({
            "open":   closes * 0.999,
            "high":   closes + np.array(spreads),
            "low":    closes - np.array(spreads),
            "close":  closes,
            "volume": [1000.0] * (n_low + n_high),
        })
        result = VolatilityAnalyzer().analyze(df)
        assert result["volatility_regime"] == "extreme"

    def test_low_regime_when_atr_shrinks(self):
        # First 44 bars high ATR, last 16 bars 10× smaller → ratio << 0.5 → low
        n_high, n_low = 44, 16
        closes = np.linspace(90000.0, 91000.0, n_high + n_low)
        spread_high = 5000.0
        spread_low = 50.0
        spreads = [spread_high] * n_high + [spread_low] * n_low
        df = pd.DataFrame({
            "open":   closes * 0.999,
            "high":   closes + np.array(spreads),
            "low":    closes - np.array(spreads),
            "close":  closes,
            "volume": [1000.0] * (n_high + n_low),
        })
        result = VolatilityAnalyzer().analyze(df)
        assert result["volatility_regime"] == "low"

    def test_insufficient_data_returns_normal(self):
        result = VolatilityAnalyzer().analyze(make_df(n=10))
        assert result["volatility_regime"] == "normal"

    def test_none_returns_normal(self):
        result = VolatilityAnalyzer().analyze(None)
        assert result["volatility_regime"] == "normal"
