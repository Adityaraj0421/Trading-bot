# tests/test_trigger_engine.py
"""Tests for OrderFlowTrigger and TriggerEngine."""

from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd

from decision import TriggerSignal
from trigger_engine import TriggerEngine
from triggers.orderflow import OrderFlowTrigger

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def make_neutral_orderflow(n_prices: int = 10) -> dict:
    """Neutral orderflow: no divergence, imbalance near mean."""
    prices = [90000.0 + i * 10 for i in range(n_prices)]  # gentle rise
    cvd = [float(i) for i in range(n_prices)]             # CVD confirming rise
    ratio_history = [1.0] * 25                             # tight history
    return {
        "prices": prices,
        "cvd": cvd,
        "imbalance_ratio": 1.0,
        "ratio_history": ratio_history,
    }


def make_1h_df_with_signals(n: int = 50) -> pd.DataFrame:
    """V-shape that produces a MomentumTrigger long signal at bar 49."""
    decline = np.linspace(95000.0, 88000.0, 40)
    partial_recovery = np.linspace(88000.0, 88700.0, 9)
    spike = np.array([91000.0])
    full_closes = np.concatenate([decline, partial_recovery, spike])
    closes = full_closes[:n]
    volumes = ([1000.0] * 44 + [3000.0] * 6)[:n]
    return pd.DataFrame(
        {
            "open": closes * 0.999,
            "high": closes * 1.003,
            "low": closes * 0.997,
            "close": closes,
            "volume": volumes,
        }
    )


# ---------------------------------------------------------------------------
# TestOrderFlowTrigger
# ---------------------------------------------------------------------------


class TestOrderFlowTrigger:
    def test_returns_list(self):
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        data = make_neutral_orderflow()
        result = trigger.evaluate(**data)
        assert isinstance(result, list)

    def test_cvd_divergence_bearish_produces_short(self):
        """Price new high but CVD not confirming → exhaustion → short."""
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        prices = [90000.0, 90500.0, 91000.0, 91500.0]  # rising price
        cvd = [100.0, 100.0, 100.0, 95.0]               # CVD falling — non-confirming
        signals = trigger.evaluate(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 10,  # too short for imbalance check
        )
        directions = [s.direction for s in signals]
        assert "short" in directions

    def test_cvd_divergence_bullish_produces_long(self):
        """Price new low but CVD not confirming → exhaustion → long."""
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        prices = [91500.0, 91000.0, 90500.0, 90000.0]  # falling price
        cvd = [100.0, 105.0, 110.0, 115.0]              # CVD rising — non-confirming
        signals = trigger.evaluate(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 10,
        )
        directions = [s.direction for s in signals]
        assert "long" in directions

    def test_imbalance_spike_produces_long(self):
        """Bid-heavy imbalance > 2σ → long signal."""
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        # Build a history with real std so spike is meaningful
        base_history = [1.0 + 0.05 * (i % 5) for i in range(25)]  # std ≈ 0.07
        current_ratio = 2.0  # well above mean + 2σ

        signals = trigger.evaluate(
            prices=[90000.0] * 5,
            cvd=[100.0] * 5,            # no divergence (flat)
            imbalance_ratio=current_ratio,
            ratio_history=base_history,
        )
        directions = [s.direction for s in signals]
        assert "long" in directions

    def test_imbalance_spike_produces_short(self):
        """Ask-heavy imbalance < 2σ → short signal."""
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        base_history = [1.0 + 0.05 * (i % 5) for i in range(25)]  # mean≈1.1, std≈0.07
        current_ratio = 0.1  # well below mean - 2σ

        signals = trigger.evaluate(
            prices=[90000.0] * 5,
            cvd=[100.0] * 5,
            imbalance_ratio=current_ratio,
            ratio_history=base_history,
        )
        directions = [s.direction for s in signals]
        assert "short" in directions

    def test_neutral_conditions_produce_no_signals(self):
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        data = make_neutral_orderflow()
        signals = trigger.evaluate(**data)
        assert signals == []

    def test_symbol_scope_set_correctly(self):
        trigger = OrderFlowTrigger(symbol="ETH/USDT")
        prices = [91500.0, 91000.0, 90500.0, 90000.0]
        cvd = [100.0, 105.0, 110.0, 115.0]
        signals = trigger.evaluate(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 5,
        )
        assert all(s.symbol_scope == "ETH" for s in signals)

    def test_insufficient_prices_produces_no_cvd_signal(self):
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        signals = trigger.evaluate(
            prices=[90000.0, 91000.0],  # only 2 values — less than _CVD_LOOKBACK+1=4
            cvd=[100.0, 95.0],
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 5,
        )
        assert signals == []

    def test_signals_have_future_expiry(self):
        trigger = OrderFlowTrigger(symbol="BTC/USDT")
        prices = [91500.0, 91000.0, 90500.0, 90000.0]
        cvd = [100.0, 105.0, 110.0, 115.0]
        signals = trigger.evaluate(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 5,
        )
        for s in signals:
            assert s.expires_at > datetime.now(UTC)


# ---------------------------------------------------------------------------
# TestTriggerEngine
# ---------------------------------------------------------------------------


class TestTriggerEngine:
    def test_on_1h_close_returns_signals(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        df = make_1h_df_with_signals()
        signals = engine.on_1h_close(df)
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_on_orderflow_update_returns_signals(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        prices = [91500.0, 91000.0, 90500.0, 90000.0]
        cvd = [100.0, 105.0, 110.0, 115.0]
        signals = engine.on_orderflow_update(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 5,
        )
        assert isinstance(signals, list)
        assert len(signals) > 0

    def test_valid_signals_excludes_expired(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        past = datetime.now(UTC) - timedelta(minutes=5)
        expired = TriggerSignal(
            trigger_id="test-expired",
            source="test",
            direction="long",
            strength=0.7,
            urgency="normal",
            symbol_scope="BTC",
            reason="expired test signal",
            expires_at=past,
        )
        engine._buffer.append(expired)
        assert expired not in engine.valid_signals()

    def test_valid_signals_includes_unexpired(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        future = datetime.now(UTC) + timedelta(minutes=30)
        fresh = TriggerSignal(
            trigger_id="test-fresh",
            source="test",
            direction="long",
            strength=0.7,
            urgency="normal",
            symbol_scope="BTC",
            reason="fresh test signal",
            expires_at=future,
        )
        engine._buffer.append(fresh)
        assert fresh in engine.valid_signals()

    def test_valid_signals_accumulates_across_calls(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        df = make_1h_df_with_signals()

        engine.on_1h_close(df)
        prices = [91500.0, 91000.0, 90500.0, 90000.0]
        cvd = [100.0, 105.0, 110.0, 115.0]
        engine.on_orderflow_update(
            prices=prices,
            cvd=cvd,
            imbalance_ratio=1.0,
            ratio_history=[1.0] * 5,
        )
        assert len(engine.valid_signals()) >= 2

    def test_clear_empties_buffer(self):
        engine = TriggerEngine(symbol="BTC/USDT")
        df = make_1h_df_with_signals()
        engine.on_1h_close(df)
        assert len(engine.valid_signals()) > 0
        engine.clear()
        assert engine.valid_signals() == []

    def test_symbol_propagated_to_triggers(self):
        engine = TriggerEngine(symbol="SOL/USDT")
        df = make_1h_df_with_signals()
        signals = engine.on_1h_close(df)
        assert all(s.symbol_scope == "SOL" for s in signals)
