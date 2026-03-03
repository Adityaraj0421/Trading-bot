"""
Unit tests for strategies.py — 6 trading strategies + StrategyEngine.

Each test constructs a synthetic DataFrame with the exact indicator columns
a strategy needs, then verifies the signal, confidence, and strategy_name.
"""

import pandas as pd
import pytest

from regime_detector import MarketRegime
from sentiment import SentimentLevel, SentimentState
from strategies import (
    BreakoutStrategy,
    GridStrategy,
    IchimokuStrategy,
    MeanReversionStrategy,
    MomentumStrategy,
    ScalpingStrategy,
    SentimentDrivenStrategy,
    StrategyEngine,
    StrategySignal,
)

# ---------------------------------------------------------------------------
# DataFrame helpers
# ---------------------------------------------------------------------------


def make_df(rows: int = 25, **overrides) -> pd.DataFrame:
    """
    Build a synthetic OHLCV + indicators DataFrame.
    Default: neutral market (no signals triggered).
    Pass keyword overrides for the LAST row's indicator values.
    """
    base_price = 50000.0
    data = {
        "open": [base_price] * rows,
        "high": [base_price + 100] * rows,
        "low": [base_price - 100] * rows,
        "close": [base_price] * rows,
        "volume": [1000.0] * rows,
        "sma_20": [base_price] * rows,
        "sma_50": [base_price] * rows,
        "macd": [0.0] * rows,
        "macd_signal": [0.0] * rows,
        "rsi": [50.0] * rows,
        "volume_ratio": [1.0] * rows,
        "atr_pct": [0.02] * rows,
        "bb_position": [0.5] * rows,
        "bb_width": [0.04] * rows,
        "stoch_k": [50.0] * rows,
        "close_to_sma20": [0.0] * rows,
    }
    df = pd.DataFrame(data)
    # Apply overrides to last row
    for col, val in overrides.items():
        if col in df.columns:
            df.loc[df.index[-1], col] = val
    return df


def make_sentiment(fg: int = 50, composite: float = 0.0, contrarian: str = "HOLD") -> SentimentState:
    if fg <= 15:
        label = SentimentLevel.EXTREME_FEAR
    elif fg <= 30:
        label = SentimentLevel.FEAR
    elif fg >= 85:
        label = SentimentLevel.EXTREME_GREED
    elif fg >= 70:
        label = SentimentLevel.GREED
    else:
        label = SentimentLevel.NEUTRAL

    return SentimentState(
        fear_greed_index=fg,
        fear_greed_label=label,
        volume_sentiment=0.0,
        price_momentum_score=0.0,
        composite_score=composite,
        contrarian_signal=contrarian,
        source="test",
    )


# ---------------------------------------------------------------------------
# StrategySignal dataclass
# ---------------------------------------------------------------------------


class TestStrategySignal:
    def test_defaults(self):
        sig = StrategySignal(
            signal="BUY",
            confidence=0.7,
            strategy_name="Test",
            reason="Test reason",
        )
        assert sig.suggested_sl_pct == 0.02
        assert sig.suggested_tp_pct == 0.05


# ---------------------------------------------------------------------------
# MomentumStrategy
# ---------------------------------------------------------------------------


class TestMomentumStrategy:
    @pytest.fixture()
    def strat(self):
        return MomentumStrategy()

    def test_name_and_regimes(self, strat):
        assert strat.name == "Momentum"
        assert MarketRegime.TRENDING_UP in strat.best_regimes

    def test_bullish_alignment_generates_buy(self, strat):
        """MA bullish + MACD crossover + RSI in range + volume → BUY."""
        df = make_df()
        # Set up bullish conditions on last row
        df.loc[df.index[-1], "close"] = 51000
        df.loc[df.index[-1], "sma_20"] = 50500
        df.loc[df.index[-1], "sma_50"] = 50000
        df.loc[df.index[-1], "macd"] = 10
        df.loc[df.index[-1], "macd_signal"] = 5
        df.loc[df.index[-1], "rsi"] = 55
        df.loc[df.index[-1], "volume_ratio"] = 1.5
        # Previous row: MACD was below signal (crossover)
        df.loc[df.index[-2], "macd"] = 4
        df.loc[df.index[-2], "macd_signal"] = 5

        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence > 0.5
        assert sig.strategy_name == "Momentum"

    def test_bearish_alignment_generates_sell(self, strat):
        """MA bearish + MACD cross down + RSI in range + volume → SELL."""
        df = make_df()
        df.loc[df.index[-1], "close"] = 49000
        df.loc[df.index[-1], "sma_20"] = 49500
        df.loc[df.index[-1], "sma_50"] = 50000
        df.loc[df.index[-1], "macd"] = -10
        df.loc[df.index[-1], "macd_signal"] = -5
        df.loc[df.index[-1], "rsi"] = 45
        df.loc[df.index[-1], "volume_ratio"] = 1.5
        df.loc[df.index[-2], "macd"] = -4
        df.loc[df.index[-2], "macd_signal"] = -5

        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence > 0.5

    def test_neutral_market_generates_hold(self, strat):
        """No clear alignment → HOLD."""
        df = make_df()  # All neutral defaults
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"

    def test_uses_atr_for_stops(self, strat):
        """BUY signal should use atr_pct for SL/TP."""
        df = make_df()
        df.loc[df.index[-1], "close"] = 51000
        df.loc[df.index[-1], "sma_20"] = 50500
        df.loc[df.index[-1], "sma_50"] = 50000
        df.loc[df.index[-1], "macd"] = 10
        df.loc[df.index[-1], "macd_signal"] = 5
        df.loc[df.index[-1], "rsi"] = 55
        df.loc[df.index[-1], "volume_ratio"] = 1.5
        df.loc[df.index[-1], "atr_pct"] = 0.03
        df.loc[df.index[-2], "macd"] = 4
        df.loc[df.index[-2], "macd_signal"] = 5

        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.suggested_sl_pct == pytest.approx(0.06)  # atr_pct * 2
        assert sig.suggested_tp_pct == pytest.approx(0.09)  # atr_pct * 3


# ---------------------------------------------------------------------------
# MeanReversionStrategy
# ---------------------------------------------------------------------------


class TestMeanReversionStrategy:
    @pytest.fixture()
    def strat(self):
        return MeanReversionStrategy()

    def test_name_and_regimes(self, strat):
        assert strat.name == "MeanReversion"
        assert MarketRegime.RANGING in strat.best_regimes

    def test_oversold_generates_buy(self, strat):
        """Price at lower BB + low RSI + low stoch → BUY."""
        df = make_df(
            bb_position=0.05,
            rsi=25,
            stoch_k=15,
            close_to_sma20=-0.03,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence > 0.4

    def test_overbought_generates_sell(self, strat):
        """Price at upper BB + high RSI + high stoch → SELL."""
        df = make_df(
            bb_position=0.95,
            rsi=75,
            stoch_k=85,
            close_to_sma20=0.03,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence > 0.4

    def test_near_mean_generates_hold(self, strat):
        """Price near SMA with normal RSI → HOLD."""
        df = make_df(bb_position=0.5, rsi=50, stoch_k=50)
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"

    def test_tight_stops(self, strat):
        """MeanReversion uses tighter stops than momentum."""
        df = make_df(bb_position=0.05, rsi=25, stoch_k=15, close_to_sma20=-0.03)
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.suggested_sl_pct == 0.015
        assert sig.suggested_tp_pct == 0.025


# ---------------------------------------------------------------------------
# BreakoutStrategy
# ---------------------------------------------------------------------------


class TestBreakoutStrategy:
    @pytest.fixture()
    def strat(self):
        return BreakoutStrategy()

    def test_name_and_regimes(self, strat):
        assert strat.name == "Breakout"
        assert MarketRegime.HIGH_VOLATILITY in strat.best_regimes

    def test_upside_breakout_with_volume(self, strat):
        """BB position > 1.0 + volume surge → BUY."""
        df = make_df(bb_position=1.1, volume_ratio=2.0, bb_width=0.06)
        # Set close at 20-bar high
        df.loc[df.index[-1], "close"] = 51000
        df.loc[df.index[-1], "high"] = 51000
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence > 0.5

    def test_downside_breakout_with_volume(self, strat):
        """BB position < 0 + volume surge → SELL."""
        df = make_df(bb_position=-0.1, volume_ratio=2.0, bb_width=0.06)
        df.loc[df.index[-1], "close"] = 49000
        df.loc[df.index[-1], "low"] = 49000
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence > 0.5

    def test_no_breakout_generates_hold(self, strat):
        """Normal BB position + normal volume → HOLD."""
        df = make_df(bb_position=0.5, volume_ratio=0.9)
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"


# ---------------------------------------------------------------------------
# GridStrategy
# ---------------------------------------------------------------------------


class TestGridStrategy:
    @pytest.fixture()
    def strat(self):
        return GridStrategy(grid_spacing_pct=0.01, grid_levels=5)

    def test_name_and_regimes(self, strat):
        assert strat.name == "Grid"
        assert MarketRegime.RANGING in strat.best_regimes

    def test_below_grid_center_generates_buy(self, strat):
        """Price well below SMA20 (grid center) → BUY."""
        df = make_df()
        df.loc[df.index[-1], "close"] = 49000  # 2% below SMA20=50000
        df.loc[df.index[-1], "sma_20"] = 50000
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert "below center" in sig.reason

    def test_above_grid_center_generates_sell(self, strat):
        """Price well above SMA20 → SELL."""
        df = make_df()
        df.loc[df.index[-1], "close"] = 51000  # 2% above SMA20=50000
        df.loc[df.index[-1], "sma_20"] = 50000
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert "above center" in sig.reason

    def test_near_center_generates_hold(self, strat):
        """Price near SMA20 → HOLD."""
        df = make_df()
        df.loc[df.index[-1], "close"] = 50050
        df.loc[df.index[-1], "sma_20"] = 50000
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert "near center" in sig.reason

    def test_confidence_increases_with_deviation(self, strat):
        """Further from center = higher confidence."""
        df_close = make_df()
        df_close.loc[df_close.index[-1], "close"] = 49200  # grid_level ~-1.6
        df_close.loc[df_close.index[-1], "sma_20"] = 50000

        df_far = make_df()
        df_far.loc[df_far.index[-1], "close"] = 48500  # grid_level ~-3.0
        df_far.loc[df_far.index[-1], "sma_20"] = 50000

        sig_close = strat.generate_signal(df_close)
        sig_far = strat.generate_signal(df_far)
        # Both should BUY, but far should have higher confidence
        assert sig_close.signal == "BUY"
        assert sig_far.signal == "BUY"
        assert sig_far.confidence > sig_close.confidence

    def test_custom_grid_spacing(self):
        """Wider grid spacing needs larger deviation to trigger."""
        wide = GridStrategy(grid_spacing_pct=0.05)
        df = make_df()
        df.loc[df.index[-1], "close"] = 49200  # 1.6% below
        df.loc[df.index[-1], "sma_20"] = 50000
        sig = wide.generate_signal(df)
        # With 5% spacing, 1.6% deviation = grid_level ~-0.32 → HOLD
        assert sig.signal == "HOLD"


# ---------------------------------------------------------------------------
# ScalpingStrategy
# ---------------------------------------------------------------------------


class TestScalpingStrategy:
    @pytest.fixture()
    def strat(self):
        return ScalpingStrategy()

    def test_name_and_regimes(self, strat):
        assert strat.name == "Scalping"

    def test_insufficient_data_holds(self, strat):
        """Less than 5 rows → HOLD with 0 confidence."""
        df = make_df(rows=3)
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence == 0.0

    def test_hammer_with_volume_generates_buy(self, strat):
        """Hammer candle (long lower wick) + oversold RSI + volume → BUY."""
        df = make_df(rows=10)
        df.loc[df.index[-1], "open"] = 50000
        df.loc[df.index[-1], "close"] = 50100  # Green candle (body = 100)
        df.loc[df.index[-1], "high"] = 50120  # Tiny upper wick (20)
        df.loc[df.index[-1], "low"] = 49700  # Long lower wick (300 > body*2)
        df.loc[df.index[-1], "rsi"] = 20  # Oversold bounce
        # v9.0: volume_spike uses strict >, so 2.1 > 2.0 threshold
        df.loc[df.index[-1], "volume_ratio"] = 2.1
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence == 0.7

    def test_shooting_star_with_volume_generates_sell(self, strat):
        """Shooting star (long upper wick) + overbought RSI + volume → SELL."""
        df = make_df(rows=10)
        df.loc[df.index[-1], "open"] = 50100
        df.loc[df.index[-1], "close"] = 50000  # Red candle (body = 100)
        df.loc[df.index[-1], "high"] = 50500  # Long upper wick (400 > body*2)
        df.loc[df.index[-1], "low"] = 49980  # Tiny lower wick (20)
        df.loc[df.index[-1], "rsi"] = 80  # Overbought drop
        df.loc[df.index[-1], "volume_ratio"] = 2.1
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"

    def test_no_pattern_generates_hold(self, strat):
        """Normal candle, no extreme RSI → HOLD."""
        df = make_df(rows=10)
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"

    def test_zero_range_generates_hold(self, strat):
        """Doji with zero range → HOLD (avoids division by zero)."""
        df = make_df(rows=10)
        df.loc[df.index[-1], "open"] = 50000
        df.loc[df.index[-1], "close"] = 50000
        df.loc[df.index[-1], "high"] = 50000
        df.loc[df.index[-1], "low"] = 50000
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence == 0.0

    def test_tight_scalping_stops(self, strat):
        """Scalping uses very tight SL/TP."""
        df = make_df(rows=10)
        df.loc[df.index[-1], "open"] = 50000
        df.loc[df.index[-1], "close"] = 50100
        df.loc[df.index[-1], "high"] = 50120
        df.loc[df.index[-1], "low"] = 49700
        df.loc[df.index[-1], "rsi"] = 20
        df.loc[df.index[-1], "volume_ratio"] = 2.1  # Strict > threshold
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.suggested_sl_pct == 0.005
        assert sig.suggested_tp_pct == 0.01


# ---------------------------------------------------------------------------
# SentimentDrivenStrategy
# ---------------------------------------------------------------------------


class TestSentimentDrivenStrategy:
    @pytest.fixture()
    def strat(self):
        return SentimentDrivenStrategy()

    def test_name(self, strat):
        assert strat.name == "Sentiment"

    def test_no_sentiment_generates_hold(self, strat):
        """No SentimentState → HOLD with 0 confidence."""
        df = make_df()
        sig = strat.generate_signal(df, sentiment=None)
        assert sig.signal == "HOLD"
        assert sig.confidence == 0.0

    def test_extreme_fear_generates_buy(self, strat):
        """FG <= 15 + negative composite → contrarian BUY."""
        df = make_df()
        sentiment = make_sentiment(fg=10, composite=-0.7)
        sig = strat.generate_signal(df, sentiment)
        assert sig.signal == "BUY"
        assert sig.confidence == 0.8
        assert "Extreme fear" in sig.reason

    def test_extreme_greed_generates_sell(self, strat):
        """FG >= 85 + positive composite → contrarian SELL."""
        df = make_df()
        sentiment = make_sentiment(fg=90, composite=0.7)
        sig = strat.generate_signal(df, sentiment)
        assert sig.signal == "SELL"
        assert sig.confidence == 0.8
        assert "Extreme greed" in sig.reason

    def test_moderate_fear_generates_cautious_buy(self, strat):
        """FG <= fear_threshold(25) + negative composite → cautious BUY."""
        df = make_df()
        # v9.0: composite_threshold=0.3 uses strict <, so -0.35 < -0.3
        sentiment = make_sentiment(fg=22, composite=-0.35)
        sig = strat.generate_signal(df, sentiment)
        assert sig.signal == "BUY"
        assert sig.confidence == 0.6
        assert "Fear zone" in sig.reason

    def test_moderate_greed_generates_cautious_sell(self, strat):
        """FG >= greed_threshold(75) + positive composite → cautious SELL."""
        df = make_df()
        # v9.0: composite_threshold=0.3 uses strict >, so 0.35 > 0.3
        sentiment = make_sentiment(fg=78, composite=0.35)
        sig = strat.generate_signal(df, sentiment)
        assert sig.signal == "SELL"
        assert sig.confidence == 0.6
        assert "Greed zone" in sig.reason

    def test_neutral_sentiment_generates_hold(self, strat):
        """FG ~50, neutral composite → HOLD."""
        df = make_df()
        sentiment = make_sentiment(fg=50, composite=0.0)
        sig = strat.generate_signal(df, sentiment)
        assert sig.signal == "HOLD"
        assert "Neutral" in sig.reason

    def test_wider_stops_at_extremes(self, strat):
        """Extreme signals use wider stops (3%/8%) vs moderate (2%/5%)."""
        df = make_df()
        extreme = make_sentiment(fg=10, composite=-0.7)
        moderate = make_sentiment(fg=25, composite=-0.3)

        sig_extreme = strat.generate_signal(df, extreme)
        sig_moderate = strat.generate_signal(df, moderate)

        assert sig_extreme.suggested_sl_pct == 0.03
        assert sig_extreme.suggested_tp_pct == 0.08
        assert sig_moderate.suggested_sl_pct == 0.02
        assert sig_moderate.suggested_tp_pct == 0.05


# ---------------------------------------------------------------------------
# StrategyEngine (ensemble)
# ---------------------------------------------------------------------------


class TestStrategyEngine:
    @pytest.fixture()
    def engine(self):
        return StrategyEngine()

    def test_has_all_ten_strategies(self, engine):
        """v10: 11 strategies (was 10; added RSIDivergence)."""
        assert len(engine.strategies) == 11
        expected = {
            "Momentum",
            "MeanReversion",
            "Breakout",
            "Grid",
            "Scalping",
            "Sentiment",
            "VWAP",
            "OBVDivergence",
            "EMACrossover",
            "Ichimoku",
            "RSIDivergence",
        }
        assert set(engine.strategies.keys()) == expected

    def test_regime_strategy_map_covers_all_regimes(self, engine):
        for regime in MarketRegime:
            assert regime in engine.REGIME_STRATEGY_MAP

    def test_trending_up_uses_momentum_primary(self, engine):
        config = engine.REGIME_STRATEGY_MAP[MarketRegime.TRENDING_UP]
        assert config["primary"] == "Momentum"
        assert sum(config["weights"].values()) == pytest.approx(1.0)

    def test_ranging_uses_obv_divergence_primary(self, engine):
        # MeanReversion removed from RANGING (destroyed capital in backtests 2021-2026)
        # OBVDivergence is the backtested primary for sideways crypto markets
        config = engine.REGIME_STRATEGY_MAP[MarketRegime.RANGING]
        assert config["primary"] == "OBVDivergence"

    def test_run_returns_strategy_signal(self, engine):
        """Engine always returns a StrategySignal regardless of market state."""
        df = make_df()
        sig = engine.run(df, MarketRegime.RANGING)
        assert isinstance(sig, StrategySignal)
        assert sig.signal in ("BUY", "SELL", "HOLD")
        assert 0.0 <= sig.confidence <= 1.0

    def test_run_populates_last_signals(self, engine):
        """After run(), last_signals should have entries from active strategies."""
        df = make_df()
        engine.run(df, MarketRegime.TRENDING_UP)
        assert len(engine.last_signals) >= 1  # At least primary ran

    def test_strong_primary_short_circuits(self, engine):
        """If primary gives confidence > 0.8, secondaries are skipped."""
        df = make_df()
        # Set up strong bullish momentum
        df.loc[df.index[-1], "close"] = 51000
        df.loc[df.index[-1], "sma_20"] = 50500
        df.loc[df.index[-1], "sma_50"] = 50000
        df.loc[df.index[-1], "macd"] = 15
        df.loc[df.index[-1], "macd_signal"] = 5
        df.loc[df.index[-1], "rsi"] = 55
        df.loc[df.index[-1], "volume_ratio"] = 2.0
        df.loc[df.index[-2], "macd"] = 4
        df.loc[df.index[-2], "macd_signal"] = 5

        engine.run(df, MarketRegime.TRENDING_UP)
        # With short-circuit, only primary should be in last_signals
        assert "Momentum" in engine.last_signals
        # If confidence was > 0.8, only 1 signal recorded (short-circuited)
        primary_sig = engine.last_signals["Momentum"]
        if primary_sig.confidence > 0.8:
            assert len(engine.last_signals) == 1

    def test_neutral_market_gets_hold(self, engine):
        """Neutral indicators → HOLD from ensemble."""
        df = make_df()  # All defaults = neutral
        sig = engine.run(df, MarketRegime.RANGING)
        assert sig.signal == "HOLD"

    def test_ranging_is_no_trade_zone(self, engine):
        """RANGING regime always returns HOLD — no strategies run, no signals recorded.

        Phase 7: RANGING is an unconditional no-trade zone. 6 phases of walk-forward
        backtesting showed fee drag exceeds signal edge in sideways markets regardless
        of strategy combination.
        """
        sig = engine.run(df=make_df(), regime=MarketRegime.RANGING)

        assert sig.signal == "HOLD"
        assert sig.strategy_name == "Ensemble"
        assert "RANGING" in sig.reason
        # No strategies run — last_signals is empty
        assert engine.last_signals == {}

    def test_unknown_regime_falls_back_to_ranging(self, engine):
        """If regime isn't in map, defaults to RANGING config."""
        # All regimes are in the map, but verify the .get() fallback path
        config = engine.REGIME_STRATEGY_MAP.get("NONEXISTENT", engine.REGIME_STRATEGY_MAP[MarketRegime.RANGING])
        assert config["primary"] == "OBVDivergence"  # current RANGING primary

    def test_weights_sum_to_one(self, engine):
        """All regime weight configs should sum to ~1.0."""
        for regime, config in engine.REGIME_STRATEGY_MAP.items():
            total = sum(config["weights"].values())
            assert total == pytest.approx(1.0), f"{regime}: weights sum to {total}"

    def test_ichimoku_in_trending_regimes(self, engine):
        """Ichimoku should appear as a secondary in TRENDING_UP and TRENDING_DOWN."""
        from regime_detector import MarketRegime

        up_secondaries = engine.REGIME_STRATEGY_MAP[MarketRegime.TRENDING_UP]["secondary"]
        down_secondaries = engine.REGIME_STRATEGY_MAP[MarketRegime.TRENDING_DOWN]["secondary"]
        assert "Ichimoku" in up_secondaries
        assert "Ichimoku" in down_secondaries


# ---------------------------------------------------------------------------
# IchimokuStrategy
# ---------------------------------------------------------------------------


def make_ichimoku_df(rows: int = 25, **overrides) -> pd.DataFrame:
    """Build DataFrame with Ichimoku indicator columns for strategy tests.

    Default: ADX below threshold (no signal), no TK cross, price in-cloud.
    Use overrides to set specific values on the last row.
    """
    df = make_df(rows=rows)
    # Add Ichimoku-specific columns (neutral defaults)
    df["ichimoku_tk_cross"] = 0
    df["ichimoku_above_cloud"] = 0
    df["ichimoku_below_cloud"] = 0
    df["ichimoku_tenkan"] = 50000.0
    df["ichimoku_kijun"] = 50000.0
    df["adx"] = 15.0  # Below default adx_threshold (20) — no signal
    # Apply overrides to the last row
    for col, val in overrides.items():
        df.loc[df.index[-1], col] = val
    return df


class TestIchimokuStrategy:
    @pytest.fixture()
    def strat(self):
        return IchimokuStrategy()

    def test_name_and_regimes(self, strat):
        from regime_detector import MarketRegime

        assert strat.name == "Ichimoku"
        assert MarketRegime.TRENDING_UP in strat.best_regimes
        assert MarketRegime.TRENDING_DOWN in strat.best_regimes

    def test_bullish_tk_cross_above_cloud_generates_buy(self, strat):
        """TK bullish cross (=1) + above cloud + ADX >= 20 → BUY."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=1,
            ichimoku_above_cloud=1,
            adx=25.0,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence >= 0.60
        assert sig.strategy_name == "Ichimoku"
        assert "cross" in sig.reason.lower()

    def test_bearish_tk_cross_below_cloud_generates_sell(self, strat):
        """TK bearish cross (=-1) + below cloud + ADX >= 20 → SELL."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=-1,
            ichimoku_below_cloud=1,
            adx=25.0,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence >= 0.60
        assert sig.strategy_name == "Ichimoku"

    def test_weak_adx_generates_hold(self, strat):
        """ADX below threshold filters out TK cross — HOLD."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=1,
            ichimoku_above_cloud=1,
            adx=10.0,  # < threshold 20
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence < 0.5  # Not a trading signal

    def test_bullish_cross_below_cloud_generates_hold(self, strat):
        """Bullish TK cross but price below cloud — not a valid setup → HOLD."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=1,
            ichimoku_above_cloud=0,  # NOT above cloud
            ichimoku_below_cloud=1,  # actually below cloud
            adx=25.0,
        )
        sig = strat.generate_signal(df)
        # Can't get a BUY cross below cloud; no bearish cross either → HOLD
        assert sig.signal == "HOLD"

    def test_sustained_bullish_alignment_low_conf_buy(self, strat):
        """No fresh cross but tenkan > kijun + above cloud + strong ADX → BUY at 0.45."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=0,  # No fresh cross
            ichimoku_above_cloud=1,
            ichimoku_tenkan=51000.0,  # tenkan > kijun
            ichimoku_kijun=50000.0,
            adx=25.0,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence == pytest.approx(0.45)

    def test_sustained_bearish_alignment_low_conf_sell(self, strat):
        """No fresh cross but tenkan < kijun + below cloud + strong ADX → SELL at 0.45."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=0,
            ichimoku_below_cloud=1,
            ichimoku_tenkan=49000.0,  # tenkan < kijun
            ichimoku_kijun=50000.0,
            adx=25.0,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence == pytest.approx(0.45)

    def test_sl_tp_based_on_atr(self, strat):
        """SL = atr_pct × 2, TP = atr_pct × 4."""
        df = make_ichimoku_df(
            ichimoku_tk_cross=1,
            ichimoku_above_cloud=1,
            adx=25.0,
            atr_pct=0.03,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.suggested_sl_pct == pytest.approx(0.06)  # 0.03 × 2
        assert sig.suggested_tp_pct == pytest.approx(0.12)  # 0.03 × 4

    def test_volume_bonus_increases_confidence(self, strat):
        """vol_ratio > 1.2 adds 0.10 to confidence."""
        df_low_vol = make_ichimoku_df(
            ichimoku_tk_cross=1, ichimoku_above_cloud=1, adx=20.0, volume_ratio=1.0
        )
        df_high_vol = make_ichimoku_df(
            ichimoku_tk_cross=1, ichimoku_above_cloud=1, adx=20.0, volume_ratio=1.5
        )
        sig_low = strat.generate_signal(df_low_vol)
        sig_high = strat.generate_signal(df_high_vol)
        assert sig_high.confidence > sig_low.confidence

    def test_missing_ichimoku_columns_generates_hold(self, strat):
        """Strategy returns HOLD gracefully when Ichimoku columns are absent."""
        df = make_df()  # No ichimoku columns — all defaults → adx=0 < threshold=20
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence < 0.5  # Not a tradeable signal

    def test_insufficient_data_returns_hold(self, strat):
        """Fewer than 3 rows → HOLD with 0 confidence."""
        df = make_ichimoku_df(rows=2)
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence == 0.0
