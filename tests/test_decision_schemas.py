# tests/test_decision_schemas.py
from datetime import UTC, datetime, timedelta

import pytest

from decision import ContextState, Decision, TriggerSignal


def make_context(**overrides):
    defaults = dict(
        context_id="2026-03-03T14:15Z",
        swing_bias="bullish",
        allowed_directions=["long"],
        volatility_regime="normal",
        funding_pressure="neutral",
        whale_flow="neutral",
        oi_trend="neutral",
        key_levels={"support": 90000.0, "resistance": 95000.0, "poc": 92000.0},
        risk_mode="normal",
        confidence=0.75,
        tradeable=True,
        valid_until=datetime.now(UTC) + timedelta(minutes=15),
        updated_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return ContextState(**defaults)


def make_trigger(**overrides):
    defaults = dict(
        trigger_id="abc-123",
        source="momentum_1h",
        direction="long",
        strength=0.7,
        urgency="normal",
        symbol_scope="BTC",
        reason="RSI crossed 50 upward + volume 1.8x",
        expires_at=datetime.now(UTC) + timedelta(minutes=30),
        raw_data={"rsi": 52.0, "volume_ratio": 1.8},
    )
    defaults.update(overrides)
    return TriggerSignal(**defaults)


class TestContextState:
    def test_valid_construction(self):
        ctx = make_context()
        assert ctx.swing_bias == "bullish"
        assert ctx.tradeable is True
        assert ctx.confidence == 0.75

    def test_invalid_swing_bias_rejected(self):
        with pytest.raises((ValueError, TypeError)):
            make_context(swing_bias="sideways")

    def test_allowed_directions_can_be_empty(self):
        ctx = make_context(allowed_directions=[])
        assert ctx.allowed_directions == []

    def test_tradeable_is_bool(self):
        ctx = make_context(tradeable=False)
        assert ctx.tradeable is False


class TestTriggerSignal:
    def test_valid_construction(self):
        t = make_trigger()
        assert t.source == "momentum_1h"
        assert t.strength == 0.7

    def test_symbol_scope_market(self):
        t = make_trigger(symbol_scope="market")
        assert t.symbol_scope == "market"

    def test_invalid_direction_rejected(self):
        with pytest.raises(ValueError):
            make_trigger(direction="sideways")

    def test_invalid_strength_rejected(self):
        with pytest.raises(ValueError):
            make_trigger(strength=1.5)

    def test_is_expired_false_for_future(self):
        t = make_trigger()
        assert t.is_expired() is False

    def test_is_expired_true_for_past(self):
        t = make_trigger(expires_at=datetime.now(UTC) - timedelta(minutes=1))
        assert t.is_expired() is True


class TestDecision:
    def test_reject_is_frozen(self):
        d = Decision(action="reject", reason="context_not_tradeable")
        with pytest.raises((AttributeError, TypeError)):
            d.reason = "mutated"

    def test_trade_decision(self):
        d = Decision(action="trade", direction="long", route="spot", score=0.63, reason="ok")
        assert d.action == "trade"
        assert d.route == "spot"

    def test_reject_has_no_direction(self):
        d = Decision(action="reject", reason="no_context")
        assert d.direction is None
        assert d.score is None
