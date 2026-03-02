"""Tests for RSIDivergenceStrategy."""

from __future__ import annotations

import pandas as pd
import pytest

from strategies import RSIDivergenceStrategy, StrategyEngine


def _make_df(
    n: int = 35,
    close_values: list[float] | None = None,
    rsi_values: list[float] | None = None,
    stoch_k: float = 50.0,
    stoch_d: float = 50.0,
    prev_stoch_k: float = 50.0,
    prev_stoch_d: float = 50.0,
) -> pd.DataFrame:
    """Build a minimal indicator DataFrame for strategy testing."""
    if close_values is None:
        close_values = [100.0 + i * 0.1 for i in range(n)]
    if rsi_values is None:
        rsi_values = [50.0] * n

    assert len(close_values) == n
    assert len(rsi_values) == n

    # Build stoch_k/stoch_d arrays: last bar = stoch_k/stoch_d, prev bar = prev_stoch_k/prev_stoch_d
    stoch_k_arr = [50.0] * n
    stoch_d_arr = [50.0] * n
    stoch_k_arr[-1] = stoch_k
    stoch_d_arr[-1] = stoch_d
    if n >= 2:
        stoch_k_arr[-2] = prev_stoch_k
        stoch_d_arr[-2] = prev_stoch_d

    return pd.DataFrame({
        "close": close_values,
        "rsi": rsi_values,
        "stoch_k": stoch_k_arr,
        "stoch_d": stoch_d_arr,
        "atr_pct": [0.02] * n,
    })


class TestRSIDivergenceStrategyBasic:
    """Basic property tests."""

    def test_name_is_rsi_divergence(self) -> None:
        assert RSIDivergenceStrategy().name == "RSIDivergence"

    def test_best_regimes_set(self) -> None:
        from strategies import MarketRegime
        strat = RSIDivergenceStrategy()
        assert MarketRegime.RANGING in strat.best_regimes
        assert MarketRegime.TRENDING_UP in strat.best_regimes

    def test_default_params(self) -> None:
        strat = RSIDivergenceStrategy()
        assert strat.lookback == 20
        assert strat.stoch_threshold == pytest.approx(35.0)

    def test_evolved_params_accepted(self) -> None:
        strat = RSIDivergenceStrategy(params={"lookback": 25, "stoch_threshold": 30.0})
        assert strat.lookback == 25
        assert strat.stoch_threshold == pytest.approx(30.0)


class TestRSIDivergenceDataGuard:
    """Data guard: too few rows."""

    def test_too_few_rows_returns_hold_confidence_zero(self) -> None:
        strat = RSIDivergenceStrategy()
        df = _make_df(n=15)  # lookback=20, need 21+
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence == 0.0


class TestBullishDivergence:
    """Bullish divergence + stochastic crossover tests."""

    def _bull_div_df(self, n: int = 35, rsi_delta: float = 15.0) -> pd.DataFrame:
        """Create df where price makes lower low but RSI makes higher low."""
        # Window (all bars except last): price low = 95.0, RSI at that low = 30.0
        # Current bar: close = 94.0 (lower low), RSI = 30.0 + rsi_delta (higher low)
        close_vals = [100.0] * (n - 1) + [94.0]
        close_vals[16] = 95.0  # the window's price low (inside lookback window, indices 14-33)
        rsi_vals = [50.0] * (n - 1) + [30.0 + rsi_delta]
        rsi_vals[16] = 30.0  # RSI at the window's price low
        return _make_df(
            n=n,
            close_values=close_vals,
            rsi_values=rsi_vals,
            stoch_k=32.0,   # oversold + crossed above stoch_d
            stoch_d=28.0,   # current stoch_d < stoch_k (crossover happened)
            prev_stoch_k=27.0,  # prev stoch_k <= prev stoch_d
            prev_stoch_d=30.0,
        )

    def test_bullish_divergence_with_stoch_confirm_generates_buy(self) -> None:
        strat = RSIDivergenceStrategy()
        df = self._bull_div_df(rsi_delta=5.0)  # weak divergence
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence == pytest.approx(0.55)

    def test_strong_bullish_divergence_higher_confidence(self) -> None:
        strat = RSIDivergenceStrategy()
        df = self._bull_div_df(rsi_delta=15.0)  # strong divergence
        sig = strat.generate_signal(df)
        assert sig.signal == "BUY"
        assert sig.confidence == pytest.approx(0.65)

    def test_divergence_without_stoch_crossover_is_hold(self) -> None:
        """Bullish divergence present but no stoch crossover -> HOLD."""
        strat = RSIDivergenceStrategy()
        df = self._bull_div_df(rsi_delta=15.0)
        # Override: stoch_k was already ABOVE stoch_d on prev bar (no crossover)
        df = df.copy()
        df.loc[df.index[-2], "stoch_k"] = 35.0
        df.loc[df.index[-2], "stoch_d"] = 30.0
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"

    def test_stoch_crossover_not_in_oversold_zone_is_hold(self) -> None:
        """Bullish divergence + stoch cross up but not oversold -> HOLD."""
        strat = RSIDivergenceStrategy()
        close_vals = [100.0] * 34 + [94.0]
        close_vals[16] = 95.0
        rsi_vals = [50.0] * 34 + [45.0]
        rsi_vals[16] = 30.0
        # stoch_k = 60 (NOT oversold, threshold=35)
        df = _make_df(
            n=35,
            close_values=close_vals,
            rsi_values=rsi_vals,
            stoch_k=60.0,
            stoch_d=58.0,
            prev_stoch_k=55.0,
            prev_stoch_d=58.0,
        )
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"


class TestBearishDivergence:
    """Bearish divergence + stochastic crossover tests."""

    def _bear_div_df(self, n: int = 35, rsi_delta: float = 15.0) -> pd.DataFrame:
        """Create df where price makes higher high but RSI makes lower high."""
        close_vals = [100.0] * (n - 1) + [106.0]
        close_vals[16] = 105.0  # window's price high (inside lookback window, indices 14-33)
        rsi_vals = [50.0] * (n - 1) + [70.0 - rsi_delta]
        rsi_vals[16] = 70.0  # RSI at window's price high
        return _make_df(
            n=n,
            close_values=close_vals,
            rsi_values=rsi_vals,
            stoch_k=70.0,   # overbought
            stoch_d=72.0,
            prev_stoch_k=72.0,  # prev >= prev_d
            prev_stoch_d=70.0,
        )

    def test_bearish_divergence_with_stoch_confirm_generates_sell(self) -> None:
        strat = RSIDivergenceStrategy()
        df = self._bear_div_df(rsi_delta=5.0)
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence == pytest.approx(0.55)

    def test_strong_bearish_divergence_higher_confidence(self) -> None:
        strat = RSIDivergenceStrategy()
        df = self._bear_div_df(rsi_delta=15.0)
        sig = strat.generate_signal(df)
        assert sig.signal == "SELL"
        assert sig.confidence == pytest.approx(0.65)


class TestHoldBehavior:
    """HOLD signal tests."""

    def test_neutral_df_returns_hold(self) -> None:
        strat = RSIDivergenceStrategy()
        df = _make_df(n=35)  # flat price/RSI, no divergence
        sig = strat.generate_signal(df)
        assert sig.signal == "HOLD"
        assert sig.confidence == pytest.approx(0.3)

    def test_hold_has_strategy_name(self) -> None:
        strat = RSIDivergenceStrategy()
        df = _make_df(n=35)
        sig = strat.generate_signal(df)
        assert sig.strategy_name == "RSIDivergence"


class TestDivergenceHelpers:
    """Unit tests for divergence detection helpers."""

    def test_detect_bullish_returns_tuple(self) -> None:
        strat = RSIDivergenceStrategy()
        df = _make_df(n=35)
        result = strat._detect_bullish_divergence(df)
        assert isinstance(result, tuple)
        assert len(result) == 2
        assert isinstance(result[0], bool)
        assert isinstance(result[1], float)

    def test_detect_bearish_returns_tuple(self) -> None:
        strat = RSIDivergenceStrategy()
        df = _make_df(n=35)
        result = strat._detect_bearish_divergence(df)
        assert isinstance(result, tuple)
        assert len(result) == 2


class TestStrategyEngineIntegration:
    """Integration with StrategyEngine."""

    def test_strategy_engine_has_11_strategies(self) -> None:
        engine = StrategyEngine()
        assert len(engine.strategies) == 11

    def test_rsi_divergence_in_engine(self) -> None:
        engine = StrategyEngine()
        assert "RSIDivergence" in engine.strategies

    def test_rsi_divergence_in_ranging_secondaries(self) -> None:
        from strategies import MarketRegime
        secondaries = StrategyEngine.REGIME_STRATEGY_MAP[MarketRegime.RANGING]["secondary"]
        assert "RSIDivergence" in secondaries
