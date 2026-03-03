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

    def test_signal_not_expired(self):
        from datetime import UTC, datetime

        from triggers.liquidity_sweep import LiquiditySweepTrigger
        trigger = LiquiditySweepTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(make_sweep_long_df())
        assert len(signals) == 1
        assert signals[0].expires_at > datetime.now(UTC)
