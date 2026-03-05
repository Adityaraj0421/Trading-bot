# tests/test_triggers.py
from datetime import UTC, datetime

import numpy as np
import pandas as pd

from triggers.momentum import MomentumTrigger


def make_bullish_1h_df(n: int = 50) -> pd.DataFrame:
    """V-shape designed so RSI crosses 50↑ and vol confirms at bar 49.

    40-bar decline (95000→88000) drives RSI below 50.
    9-bar partial recovery (88000→88700) keeps RSI just below 50 (~44).
    Final spike to 91000 pushes RSI to ~81, triggering rsi_crossed_up +
    vol_confirmed → long_score = 2.

    When n < 50, returns n bars of the declining phase only — used for the
    insufficient-data test (< _MIN_BARS = 26 triggers early return).
    """
    decline = np.linspace(95000.0, 88000.0, 40)
    partial_recovery = np.linspace(88000.0, 88700.0, 9)
    spike = np.array([91000.0])
    full_closes = np.concatenate([decline, partial_recovery, spike])
    full_volumes = [1000.0] * 44 + [3000.0] * 6

    closes = full_closes[:n]
    volumes = full_volumes[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.003,
            "low": closes * 0.997,
            "close": closes,
            "volume": volumes,
        }
    )


def make_bearish_1h_df(n: int = 50) -> pd.DataFrame:
    """Inverse V-shape designed so RSI crosses 50↓ and vol confirms at bar 49.

    40-bar rise (88000→95000) drives RSI above 50.
    9-bar partial decline (95000→94300) keeps RSI just above 50 (~56).
    Final crash to 91000 drops RSI to ~15, triggering rsi_crossed_down +
    vol_confirmed → short_score = 2.
    """
    rise = np.linspace(88000.0, 95000.0, 40)
    partial_decline = np.linspace(95000.0, 94300.0, 9)
    crash = np.array([91000.0])
    full_closes = np.concatenate([rise, partial_decline, crash])
    full_volumes = [1000.0] * 44 + [3000.0] * 6

    closes = full_closes[:n]
    volumes = full_volumes[:n]
    return pd.DataFrame(
        {
            "open": closes * 1.001,
            "high": closes * 1.003,
            "low": closes * 0.997,
            "close": closes,
            "volume": volumes,
        }
    )


def make_flat_df(n: int = 50) -> pd.DataFrame:
    """Flat prices — no momentum signal expected."""
    closes = [95000.0] * n
    return pd.DataFrame(
        {
            "open": closes,
            "high": [c * 1.001 for c in closes],
            "low": [c * 0.999 for c in closes],
            "close": closes,
            "volume": [1000.0] * n,
        }
    )


class TestMomentumTrigger:
    def test_returns_trigger_signal_list(self):

        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        assert isinstance(signals, list)

    def test_bullish_momentum_produces_long_signal(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        directions = [s.direction for s in signals]
        assert "long" in directions

    def test_bearish_momentum_produces_short_signal(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bearish_1h_df())
        directions = [s.direction for s in signals]
        assert "short" in directions

    def test_flat_market_produces_no_signals(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_flat_df())
        assert signals == []

    def test_signals_have_correct_symbol_scope(self):
        trigger = MomentumTrigger(symbol="ETH/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        for s in signals:
            assert s.symbol_scope == "ETH"

    def test_signals_have_future_expiry(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df())
        for s in signals:
            assert s.expires_at > datetime.now(UTC)

    def test_insufficient_data_returns_empty(self):
        trigger = MomentumTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_bullish_1h_df(n=5))
        assert signals == []


# ---------------------------------------------------------------------------
# TestLiquidationTrigger
# ---------------------------------------------------------------------------


class TestLiquidationTrigger:
    def test_long_liquidation_cascade_produces_short_signal(self):
        from triggers.liquidation import LiquidationTrigger

        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "long"})
        assert any(s.direction == "short" for s in signals)

    def test_short_liquidation_cascade_produces_long_signal(self):
        from triggers.liquidation import LiquidationTrigger

        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "short"})
        assert any(s.direction == "long" for s in signals)

    def test_small_liquidation_ignored(self):
        from triggers.liquidation import LiquidationTrigger

        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 100_000, "direction": "long"})
        assert signals == []

    def test_signals_have_high_urgency(self):
        from triggers.liquidation import LiquidationTrigger

        trigger = LiquidationTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate({"liq_volume_usd": 50_000_000, "direction": "long"})
        assert all(s.urgency == "high" for s in signals)

    def test_none_data_returns_empty(self):
        from triggers.liquidation import LiquidationTrigger

        trigger = LiquidationTrigger(symbol="BTC/USDT")
        assert trigger.evaluate(None) == []


# ---------------------------------------------------------------------------
# TestFundingExtremeTrigger
# ---------------------------------------------------------------------------


class TestFundingExtremeTrigger:
    def test_extreme_positive_funding_produces_short(self):
        from triggers.funding_extreme import FundingExtremeTrigger

        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0012)
        assert any(s.direction == "short" for s in signals)

    def test_extreme_negative_funding_produces_long(self):
        from triggers.funding_extreme import FundingExtremeTrigger

        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=-0.0006)
        assert any(s.direction == "long" for s in signals)

    def test_normal_funding_produces_no_signal(self):
        from triggers.funding_extreme import FundingExtremeTrigger

        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0001)
        assert signals == []

    def test_signals_have_high_urgency(self):
        from triggers.funding_extreme import FundingExtremeTrigger

        trigger = FundingExtremeTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(funding_rate=0.0012)
        assert all(s.urgency == "high" for s in signals)


# --- LiquiditySweepTrigger tests ---


def make_sweep_long_df() -> pd.DataFrame:
    """21-bar 1h DataFrame: equal lows cluster then sweep bar wicks below and closes above."""
    n = 21
    closes = [90000.0] * 10 + [89000.0] * 10 + [89500.0]  # recovery close on last bar
    lows   = [88100.0] * 10 + [88050.0] * 10 + [87500.0]  # last bar wicks below zone
    highs  = [91000.0] * 10 + [90000.0] * 10 + [90000.0]
    opens  = [c * 0.999 for c in closes]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * n,
    })


def make_sweep_short_df() -> pd.DataFrame:
    """21-bar 1h DataFrame: equal highs cluster then sweep bar wicks above and closes below."""
    n = 21
    closes = [95000.0] * 10 + [96000.0] * 10 + [95500.0]  # rejection close
    highs  = [97100.0] * 10 + [97050.0] * 10 + [97600.0]  # last bar wicks above zone
    lows   = [94000.0] * 10 + [95000.0] * 10 + [94000.0]
    opens  = [c * 1.001 for c in closes]
    return pd.DataFrame({
        "open": opens, "high": highs, "low": lows,
        "close": closes, "volume": [1000.0] * n,
    })


def make_no_sweep_df() -> pd.DataFrame:
    """Trending up bars — no equal highs/lows cluster."""
    closes = np.linspace(88000.0, 96000.0, 21)
    return pd.DataFrame({
        "open": closes * 0.999, "high": closes * 1.005,
        "low": closes * 0.995, "close": closes,
        "volume": [1000.0] * 21,
    })


class TestLiquiditySweepTrigger:
    def test_bullish_sweep_fires_long_signal(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_long_df())
        assert len(signals) == 1
        assert signals[0].direction == "long"
        assert signals[0].source == "liquidity_sweep"
        assert signals[0].strength == 0.65

    def test_bearish_sweep_fires_short_signal(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_short_df())
        assert len(signals) == 1
        assert signals[0].direction == "short"

    def test_no_sweep_returns_empty(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_no_sweep_df())
        assert signals == []

    def test_insufficient_data_returns_empty(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_no_sweep_df().head(10))
        assert signals == []

    def test_insufficient_data_boundary_returns_empty(self):
        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        # Exactly _MIN_BARS - 1 = 20 rows — must still be empty
        signals = trigger.evaluate(make_no_sweep_df().head(20))
        assert signals == []

    def test_signal_not_expired(self):
        from datetime import UTC, datetime

        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_long_df())
        assert len(signals) == 1
        assert signals[0].expires_at > datetime.now(UTC)


# ---------------------------------------------------------------------------
# PullbackTrigger fixtures + tests
# ---------------------------------------------------------------------------


def make_pullback_long_df(n: int = 39) -> pd.DataFrame:
    """Uptrend then shallow pullback then two-bar recovery — RSI ≈ 49.5 in [42,52] and rising.

    Phase 1 (30 bars, 88000→95000): sustained rally drives RSI to ~72.
    Phase 2 (6 bars, 95000→93200): shallow pullback drops RSI toward mid-zone.
    Phase 3 (2 bars, 93200→93666): gradual recovery lifts RSI into [42,52] range.
    Final bar (93666→93760): one more uptick — RSI 49.5, window_min 48.4 → recovering=True.

    Total full-length: 39 bars. When n=10, returns 10 bars (< _MIN_BARS=26) — used for
    the insufficient-data test.
    """
    phase1 = np.linspace(88000.0, 95000.0, 30)
    phase2 = np.linspace(95000.0, 93200.0, 6)
    phase3 = np.linspace(93200.0, 93200.0 * 1.005, 2)
    final = np.array([phase3[-1] * 1.001])
    full = np.concatenate([phase1, phase2, phase3, final])
    closes = full[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


def make_pullback_short_df(n: int = 41) -> pd.DataFrame:
    """Downtrend then shallow bounce then two-bar rejection — RSI ≈ 52.6 in [48,58] and declining.

    Phase 1 (30 bars, 95000→88000): sustained decline drives RSI to ~28.
    Phase 2 (8 bars, 88000→89400): shallow bounce lifts RSI toward mid-zone.
    Phase 3 (2 bars, 89400→88953): gradual fade pulls RSI into [48,58] range.
    Final bar (88953→88864): one more downtick — RSI 52.6, window_max 53.7 → declining=True.

    Total full-length: 41 bars. When n=10, returns 10 bars (< _MIN_BARS=26) — used for
    the insufficient-data test.
    """
    phase1 = np.linspace(95000.0, 88000.0, 30)
    phase2 = np.linspace(88000.0, 89400.0, 8)
    phase3 = np.linspace(89400.0, 89400.0 * 0.995, 2)
    final = np.array([phase3[-1] * 0.999])
    full = np.concatenate([phase1, phase2, phase3, final])
    closes = full[:n]
    return pd.DataFrame(
        {
            "open": closes * 1.001,
            "high": closes * 1.002,
            "low": closes * 0.998,
            "close": closes,
            "volume": [1000.0] * len(closes),
        }
    )


class TestPullbackTrigger:
    def test_insufficient_bars_returns_empty(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(n=10), swing_bias="bullish")
        assert signals == []

    def test_neutral_bias_returns_empty(self):
        """swing_bias='neutral' must suppress all signals regardless of RSI."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="neutral")
        assert signals == []

    def test_long_fires_on_bullish_pullback(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert any(s.direction == "long" for s in signals)

    def test_long_not_fired_on_bearish_bias(self):
        """Bullish pullback conditions with bearish bias → no long signal."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bearish")
        assert not any(s.direction == "long" for s in signals)

    def test_short_fires_on_bearish_pullback(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_short_df(), swing_bias="bearish")
        assert any(s.direction == "short" for s in signals)

    def test_short_not_fired_on_bullish_bias(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_short_df(), swing_bias="bullish")
        assert not any(s.direction == "short" for s in signals)

    def test_source_is_pullback_1h(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert signals, "Expected at least one signal from pullback fixture"
        assert all(s.source == "pullback_1h" for s in signals)

    def test_urgency_is_normal(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert signals, "Expected at least one signal from pullback fixture"
        assert all(s.urgency == "normal" for s in signals)

    def test_signal_not_expired(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert signals, "Expected at least one signal from pullback fixture"
        assert all(s.expires_at > datetime.now(UTC) for s in signals)

    def test_strength_in_valid_range(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert signals, "Expected at least one signal from pullback fixture"
        for s in signals:
            assert 0.0 < s.strength <= 0.72

    def test_symbol_scope_extracted(self):
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="ETH/USDT")
        signals = trigger.evaluate(make_pullback_long_df(), swing_bias="bullish")
        assert signals, "Expected at least one signal from pullback fixture"
        assert all(s.symbol_scope == "ETH" for s in signals)

    def test_flat_market_no_signal(self):
        """Flat prices → RSI is NaN → trigger returns no signals."""
        from triggers.pullback import PullbackTrigger

        trigger = PullbackTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_flat_df(n=55), swing_bias="bullish")
        assert signals == []


class TestTriggerEngineSwingBias:
    def test_on_1h_close_accepts_swing_bias_kwarg(self):
        """on_1h_close must accept swing_bias without raising."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        result = eng.on_1h_close(make_pullback_long_df(), swing_bias="bullish")
        assert isinstance(result, list)

    def test_on_1h_close_default_swing_bias_neutral(self):
        """Existing call sites (no swing_bias) must continue to work."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        result = eng.on_1h_close(make_pullback_long_df())
        assert isinstance(result, list)

    def test_pullback_signal_in_buffer_after_bullish_close(self):
        """After on_1h_close with bullish bias, valid_signals includes pullback."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        eng.on_1h_close(make_pullback_long_df(), swing_bias="bullish")
        sigs = eng.valid_signals()
        pullback_sigs = [s for s in sigs if s.source == "pullback_1h"]
        assert len(pullback_sigs) >= 1

    def test_no_pullback_signal_with_neutral_bias(self):
        """Neutral bias must not produce pullback signals in buffer."""
        from trigger_engine import TriggerEngine

        eng = TriggerEngine(symbol="BTC/USDT")
        eng.on_1h_close(make_pullback_long_df(), swing_bias="neutral")
        sigs = eng.valid_signals()
        assert not any(s.source == "pullback_1h" for s in sigs)
