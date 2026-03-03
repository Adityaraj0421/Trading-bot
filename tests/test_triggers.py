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
