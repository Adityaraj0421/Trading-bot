# tests/test_decision_schemas.py
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from decision import ContextState, Decision, TriggerSignal, evaluate


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

def make_two_triggers(direction: str = "long", urgency: str = "normal", strength: float = 0.8) -> list:
    """Return two agreeing TriggerSignals (minimum for consensus)."""
    return [
        make_trigger(trigger_id=str(uuid.uuid4()), direction=direction, urgency=urgency, strength=strength),
        make_trigger(trigger_id=str(uuid.uuid4()), direction=direction, urgency=urgency, strength=strength,
                     source="orderflow"),
    ]

class TestEvaluate:
    def test_rejects_non_tradeable_context(self):
        ctx = make_context(tradeable=False)
        d = evaluate(ctx, make_two_triggers())
        assert d.action == "reject"
        assert d.reason == "context_not_tradeable"

    def test_rejects_empty_allowed_directions(self):
        ctx = make_context(tradeable=True, allowed_directions=[])
        d = evaluate(ctx, make_two_triggers())
        assert d.action == "reject"
        assert d.reason == "no_allowed_directions"

    def test_rejects_when_all_triggers_expired(self):
        past = datetime.now(UTC) - timedelta(minutes=1)
        t1 = make_trigger(trigger_id="t1", expires_at=past)
        t2 = make_trigger(trigger_id="t2", expires_at=past)
        d = evaluate(make_context(), [t1, t2])
        assert d.action == "reject"
        assert d.reason == "no_valid_triggers"

    def test_rejects_direction_mismatch_with_context(self):
        ctx = make_context(allowed_directions=["long"])
        triggers = make_two_triggers(direction="short")
        d = evaluate(ctx, triggers)
        assert d.action == "reject"
        assert d.reason == "no_valid_triggers"

    def test_rejects_single_trigger_no_consensus(self):
        d = evaluate(make_context(), [make_trigger()])
        assert d.action == "reject"
        assert d.reason == "insufficient_directional_agreement"

    def test_rejects_two_triggers_opposite_directions(self):
        ctx = make_context(allowed_directions=["long", "short"])
        long_t = make_trigger(trigger_id="t1", direction="long")
        short_t = make_trigger(trigger_id="t2", direction="short")
        d = evaluate(ctx, [long_t, short_t])
        assert d.action == "reject"
        assert d.reason == "insufficient_directional_agreement"

    def test_rejects_score_below_threshold(self):
        # confidence=0.3, strength=0.3 → score=0.09 < 0.50
        ctx = make_context(confidence=0.3)
        triggers = make_two_triggers(strength=0.3)
        d = evaluate(ctx, triggers)
        assert d.action == "reject"
        assert "score_below_threshold" in d.reason

    def test_happy_path_spot_trade(self):
        ctx = make_context(confidence=0.8)
        d = evaluate(ctx, make_two_triggers(strength=0.8, urgency="normal"))
        assert d.action == "trade"
        assert d.route == "spot"
        assert d.direction == "long"
        assert d.score == pytest.approx(0.64)

    def test_happy_path_short_spot_trade(self):
        ctx = make_context(confidence=0.8, allowed_directions=["short"])
        d = evaluate(ctx, make_two_triggers(direction="short", strength=0.8))
        assert d.action == "trade"
        assert d.direction == "short"
        assert d.route == "spot"

    def test_routes_to_perp_for_high_urgency(self):
        ctx = make_context(confidence=0.8)
        d = evaluate(ctx, make_two_triggers(urgency="high", strength=0.8))
        assert d.action == "trade"
        assert d.route == "perp"

    def test_defensive_mode_blocks_event_routing(self):
        ctx = make_context(confidence=0.9, risk_mode="defensive")
        d = evaluate(ctx, make_two_triggers(urgency="high", strength=0.9))
        assert d.action == "reject"
        assert d.reason == "event_blocked_by_risk_mode"

    def test_defensive_mode_allows_normal_urgency(self):
        # defensive blocks perp, but normal urgency → spot is still allowed
        ctx = make_context(confidence=0.8, risk_mode="defensive")
        d = evaluate(ctx, make_two_triggers(urgency="normal", strength=0.8))
        assert d.action == "trade"
        assert d.route == "spot"

    def test_score_exactly_at_threshold_passes(self):
        # confidence=0.8, strength=0.625 → score = 0.50 exactly
        ctx = make_context(confidence=0.8)
        d = evaluate(ctx, make_two_triggers(strength=0.625))
        assert d.action == "trade"

    def test_stronger_direction_wins_tie(self):
        # 2 long triggers at 0.5, 2 short triggers at 0.9 — short should win by aggregate strength
        ctx = make_context(confidence=0.9, allowed_directions=["long", "short"])
        long_triggers = make_two_triggers(direction="long", strength=0.5)
        short_triggers = make_two_triggers(direction="short", strength=0.9)
        d = evaluate(ctx, long_triggers + short_triggers)
        assert d.action == "trade"
        assert d.direction == "short"
