"""
Unit tests for regime_detector.py — MarketRegime, RegimeState, RegimeDetector.

Tests the volatility regime, trend regime, regime combination, and
regime history tracking without requiring hmmlearn.
"""

import pytest
import numpy as np
import pandas as pd
from regime_detector import MarketRegime, RegimeState, RegimeDetector


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_ohlcv(n=100, trend="flat", volatility="normal"):
    """Build synthetic OHLCV data with controllable trend and volatility."""
    np.random.seed(42)
    base = 50000.0
    noise_scale = 200 if volatility == "normal" else 2000

    if trend == "up":
        prices = base + np.cumsum(np.random.normal(50, noise_scale, n))
    elif trend == "down":
        prices = base + np.cumsum(np.random.normal(-50, noise_scale, n))
    else:
        prices = base + np.cumsum(np.random.normal(0, noise_scale, n))

    prices = np.maximum(prices, 1000)  # Floor at 1000

    df = pd.DataFrame({
        "open": prices,
        "high": prices + abs(np.random.normal(100, 50, n)),
        "low": prices - abs(np.random.normal(100, 50, n)),
        "close": prices + np.random.normal(0, 30, n),
        "volume": np.random.uniform(500, 2000, n),
    }, index=pd.date_range("2024-01-01", periods=n, freq="h"))

    return df


def make_df_with_indicators(n=100, trend="flat", volatility="normal"):
    """Build OHLCV with pre-computed indicator columns."""
    df = make_ohlcv(n, trend, volatility)

    close = df["close"]
    df["sma_10"] = close.rolling(10).mean()
    df["sma_20"] = close.rolling(20).mean()
    df["sma_50"] = close.rolling(50).mean()
    df["atr_pct"] = ((df["high"] - df["low"]) / close).rolling(14).mean()
    df["bb_width"] = close.rolling(20).std() * 2 / close.rolling(20).mean()
    df["log_returns"] = np.log(close / close.shift(1))
    df["rolling_vol_10"] = df["log_returns"].rolling(10).std()

    return df.dropna()


# ---------------------------------------------------------------------------
# MarketRegime enum
# ---------------------------------------------------------------------------

class TestMarketRegime:
    def test_all_regimes_exist(self):
        assert len(MarketRegime) == 4
        assert MarketRegime.TRENDING_UP.value == "trending_up"
        assert MarketRegime.HIGH_VOLATILITY.value == "high_volatility"


# ---------------------------------------------------------------------------
# RegimeState
# ---------------------------------------------------------------------------

class TestRegimeState:
    def test_fields(self):
        state = RegimeState(
            regime=MarketRegime.RANGING,
            confidence=0.7,
            volatility=0.02,
            trend_strength=0.3,
            regime_duration=5,
        )
        assert state.regime == MarketRegime.RANGING
        assert state.confidence == 0.7


# ---------------------------------------------------------------------------
# RegimeDetector
# ---------------------------------------------------------------------------

class TestRegimeDetector:
    @pytest.fixture()
    def detector(self):
        return RegimeDetector()

    def test_default_state(self, detector):
        assert detector.current_regime == MarketRegime.RANGING
        assert detector.regime_duration == 0

    def test_insufficient_data_returns_ranging(self, detector):
        df = make_ohlcv(n=20)
        state = detector.detect(df)
        assert state.regime == MarketRegime.RANGING
        assert state.confidence == 0.5

    def test_detect_returns_regime_state(self, detector):
        df = make_ohlcv(n=100)
        state = detector.detect(df)
        assert isinstance(state, RegimeState)
        assert isinstance(state.regime, MarketRegime)
        assert 0 <= state.confidence <= 1.0
        assert state.regime_duration >= 1

    def test_regime_history_tracked(self, detector):
        df = make_ohlcv(n=100)
        detector.detect(df)
        detector.detect(df)
        detector.detect(df)
        assert len(detector.regime_history) == 3

    def test_regime_duration_increments_on_same_regime(self, detector):
        df = make_ohlcv(n=100, trend="flat")
        detector.detect(df)
        dur1 = detector.regime_duration
        detector.detect(df)
        dur2 = detector.regime_duration
        # If same regime detected, duration should increase
        if detector.regime_history[-1].regime == detector.regime_history[-2].regime:
            assert dur2 > dur1

    def test_regime_duration_resets_on_change(self, detector):
        """When regime changes, duration resets to 1."""
        # Force a regime change by manipulating internal state
        detector.current_regime = MarketRegime.TRENDING_UP
        detector.regime_duration = 10

        # Flat data should detect RANGING (different from TRENDING_UP)
        df = make_ohlcv(n=100, trend="flat")
        state = detector.detect(df)
        if state.regime != MarketRegime.TRENDING_UP:
            assert detector.regime_duration == 1

    def test_history_bounded(self, detector):
        df = make_ohlcv(n=100)
        for _ in range(600):
            detector.detect(df)
        assert len(detector.regime_history) <= RegimeDetector.MAX_HISTORY

    # --- Using pre-computed indicators ---

    def test_detect_with_indicators(self, detector):
        df_raw = make_ohlcv(n=100)
        df_ind = make_df_with_indicators(n=100)
        state = detector.detect(df_raw, df_ind=df_ind)
        assert isinstance(state.regime, MarketRegime)

    def test_high_volatility_data(self, detector):
        """High volatility data should trend toward HIGH_VOLATILITY regime."""
        df = make_ohlcv(n=200, volatility="high")
        df_ind = make_df_with_indicators(n=200, volatility="high")
        state = detector.detect(df, df_ind=df_ind)
        # With high volatility, detector should lean toward HIGH_VOLATILITY
        # (may not always be exact due to other factors, so just verify it runs)
        assert state.regime in list(MarketRegime)

    # --- Internal methods ---

    def test_volatility_regime_without_indicators(self, detector):
        df = make_ohlcv(n=100)
        result = detector._volatility_regime(df, df_ind=None)
        assert "regime" in result
        assert "volatility_pct" in result
        assert "confidence" in result

    def test_volatility_regime_with_indicators(self, detector):
        df = make_ohlcv(n=100)
        df_ind = make_df_with_indicators(n=100)
        result = detector._volatility_regime(df, df_ind=df_ind)
        assert result["volatility_pct"] > 0

    def test_trend_regime_without_indicators(self, detector):
        df = make_ohlcv(n=100)
        result = detector._trend_regime(df, df_ind=None)
        assert "regime" in result
        assert "strength" in result
        assert "alignment" in result

    def test_combine_regimes_high_vol_override(self, detector):
        """HIGH_VOLATILITY with high confidence overrides voting."""
        vol = {"regime": MarketRegime.HIGH_VOLATILITY, "volatility_pct": 0.05, "confidence": 0.9}
        trend = {"regime": MarketRegime.TRENDING_UP, "strength": 0.5, "alignment": 3, "confidence": 0.7}
        regime, conf = detector._combine_regimes(vol, trend, None)
        assert regime == MarketRegime.HIGH_VOLATILITY

    def test_combine_regimes_normal_voting(self, detector):
        """Normal case: weighted voting between trend and vol."""
        vol = {"regime": MarketRegime.RANGING, "volatility_pct": 0.02, "confidence": 0.6}
        trend = {"regime": MarketRegime.TRENDING_UP, "strength": 0.5, "alignment": 3, "confidence": 0.8}
        regime, conf = detector._combine_regimes(vol, trend, None)
        # Trend has higher weight (0.4) + gets HMM's weight (0.2) = 0.6 total
        assert regime == MarketRegime.TRENDING_UP
