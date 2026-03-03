# tests/test_decision.py
import uuid
from datetime import UTC, datetime, timedelta

from decision import ContextState, TriggerSignal, evaluate


def make_context(**overrides):
    """Build a minimal tradeable ContextState. Pass keyword overrides."""
    defaults = dict(
        context_id="2026-03-02T16:00Z",
        swing_bias="bullish",
        allowed_directions=["long"],
        volatility_regime="normal",
        funding_pressure="neutral",
        whale_flow="neutral",
        oi_trend="neutral",
        key_levels={"support": 90000.0, "resistance": 110000.0, "poc": 100000.0},
        risk_mode="normal",
        confidence=0.8,
        tradeable=True,
        valid_until=datetime.now(UTC) + timedelta(hours=1),
        updated_at=datetime.now(UTC),
    )
    defaults.update(overrides)
    return ContextState(**defaults)


def make_trigger(direction: str = "long", strength: float = 0.8, urgency: str = "normal"):
    return TriggerSignal(
        trigger_id=str(uuid.uuid4()),
        source="test",
        direction=direction,
        strength=strength,
        urgency=urgency,
        symbol_scope="BTC",
        reason="test",
        expires_at=datetime.now(UTC) + timedelta(hours=1),
    )


class TestFundingExtremeGate:
    def test_long_crowded_extreme_blocks_long_entry(self):
        ctx = make_context(funding_pressure="long_crowded_extreme")
        triggers = [make_trigger("long"), make_trigger("long")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "reject"
        assert decision.reason == "funding_extreme_blocks_direction"

    def test_short_crowded_extreme_blocks_short_entry(self):
        ctx = make_context(
            swing_bias="bearish",
            allowed_directions=["short"],
            funding_pressure="short_crowded_extreme",
        )
        triggers = [make_trigger("short"), make_trigger("short")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "reject"
        assert decision.reason == "funding_extreme_blocks_direction"

    def test_long_crowded_extreme_allows_short_entry(self):
        # Extreme long funding -> only blocks longs, not shorts
        ctx = make_context(
            swing_bias="bearish",
            allowed_directions=["short"],
            funding_pressure="long_crowded_extreme",
        )
        triggers = [make_trigger("short"), make_trigger("short")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "trade"
        assert decision.direction == "short"

    def test_long_crowded_mild_does_not_block(self):
        ctx = make_context(funding_pressure="long_crowded_mild")
        triggers = [make_trigger("long"), make_trigger("long")]
        decision = evaluate(ctx, triggers)
        assert decision.action == "trade"
