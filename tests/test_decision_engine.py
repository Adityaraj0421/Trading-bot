"""
Unit tests for decision_engine.py — state machine, safety checks, signal overrides.

Covers:
  - DecisionState transitions (NORMAL → CAUTIOUS → DEFENSIVE → HALTED)
  - Safety gating (capital floor, daily loss, consecutive losses)
  - Signal override logic per state
  - Trade result recording + consecutive loss tracking
  - Event logging
  - Serialization round-trip
"""

from datetime import datetime, timedelta

import pytest

from decision_engine import (
    AutonomousEvent,
    DecisionEngine,
    DecisionState,
)


@pytest.fixture()
def engine():
    return DecisionEngine(initial_capital=10000)


# ---------------------------------------------------------------------------
# AutonomousEvent
# ---------------------------------------------------------------------------


class TestAutonomousEvent:
    def test_to_dict_shape(self):
        ev = AutonomousEvent(event_type="test", description="Test event")
        d = ev.to_dict()
        assert d["type"] == "test"
        assert d["description"] == "Test event"
        assert "timestamp" in d
        assert "data" in d


# ---------------------------------------------------------------------------
# Safety checks / state transitions
# ---------------------------------------------------------------------------


class TestSafetyChecks:
    """Verify _check_safety and orchestrate manage state correctly."""

    def test_starts_in_normal(self, engine):
        assert engine.state == DecisionState.NORMAL

    def test_halted_when_capital_below_floor(self, engine):
        """Capital drops below 50% of initial → HALTED."""
        instructions = engine.orchestrate(cycle_count=1, current_capital=4000)
        assert engine.state == DecisionState.HALTED
        assert instructions["should_trade"] is False
        assert "HALTED" in instructions["skip_reason"]

    def test_defensive_on_daily_loss(self, engine):
        """Daily PnL exceeds max_daily_loss_pct → DEFENSIVE."""
        # Simulate big daily loss via _daily_pnl
        engine._daily_pnl = -600  # 6% of 10000 > 5% limit
        engine._daily_reset_date = datetime.now()
        instructions = engine.orchestrate(cycle_count=1, current_capital=9400)
        assert engine.state == DecisionState.DEFENSIVE
        assert instructions["position_multiplier"] == engine.config.defensive_position_mult

    def test_cautious_on_consecutive_losses(self, engine):
        """5+ consecutive losses → CAUTIOUS."""
        engine._consecutive_losses = 5
        instructions = engine.orchestrate(cycle_count=1, current_capital=9500)
        assert engine.state == DecisionState.CAUTIOUS
        assert instructions["position_multiplier"] == engine.config.cautious_position_mult

    def test_normal_trading_allowed(self, engine):
        """Normal state: full trading, multiplier=1.0."""
        instructions = engine.orchestrate(cycle_count=1, current_capital=10000)
        assert engine.state == DecisionState.NORMAL
        assert instructions["should_trade"] is True
        assert instructions["position_multiplier"] == 1.0

    def test_auto_recovery_from_cautious(self, engine):
        """After 30 min with < 3 consecutive losses, recover to NORMAL."""
        engine.state = DecisionState.CAUTIOUS
        engine._state_change_time = datetime.now() - timedelta(minutes=35)
        engine._consecutive_losses = 1  # Below recovery threshold

        engine.orchestrate(cycle_count=1, current_capital=9500)
        assert engine.state == DecisionState.NORMAL

    def test_no_recovery_if_losses_still_high(self, engine):
        """Don't recover if consecutive_losses >= 3 even after 30 min."""
        engine.state = DecisionState.CAUTIOUS
        engine._state_change_time = datetime.now() - timedelta(minutes=35)
        engine._consecutive_losses = 5  # Still too many

        engine.orchestrate(cycle_count=1, current_capital=9500)
        assert engine.state == DecisionState.CAUTIOUS

    def test_halted_takes_priority_over_defensive(self, engine):
        """Capital floor check (HALTED) takes priority over daily loss (DEFENSIVE)."""
        engine._daily_pnl = -600  # Would trigger DEFENSIVE
        engine.orchestrate(cycle_count=1, current_capital=3000)  # Below 50%
        assert engine.state == DecisionState.HALTED

    def test_state_change_logs_event(self, engine):
        """State transitions are logged as events."""
        engine.orchestrate(cycle_count=1, current_capital=4000)
        assert len(engine.event_log) > 0
        event = engine.event_log[-1]
        assert event.event_type == "state_change"
        assert "HALTED" in event.description or "halted" in event.description


# ---------------------------------------------------------------------------
# Signal overrides
# ---------------------------------------------------------------------------


class TestSignalOverride:
    """Verify should_override_signal behavior per state."""

    def test_halted_overrides_to_hold(self, engine):
        engine.state = DecisionState.HALTED
        signal, conf = engine.should_override_signal("BUY", 0.9)
        assert signal == "HOLD"
        assert conf == 0.0

    def test_defensive_blocks_low_confidence(self, engine):
        engine.state = DecisionState.DEFENSIVE
        # v7.0: DEFENSIVE blocks below 0.5 confidence (relaxed from 0.75)
        signal, conf = engine.should_override_signal("BUY", 0.4)
        assert signal == "HOLD"

    def test_defensive_dampens_high_confidence(self, engine):
        engine.state = DecisionState.DEFENSIVE
        signal, conf = engine.should_override_signal("BUY", 0.9)
        assert signal == "BUY"
        # v7.0: DEFENSIVE dampens by 0.8 (relaxed from 0.7)
        assert conf == pytest.approx(0.9 * 0.8)

    def test_cautious_dampens_confidence(self, engine):
        engine.state = DecisionState.CAUTIOUS
        signal, conf = engine.should_override_signal("SELL", 0.8)
        assert signal == "SELL"
        # v7.0: CAUTIOUS dampens by 0.9 (relaxed from 0.8)
        assert conf == pytest.approx(0.8 * 0.9)

    def test_normal_passes_through(self, engine):
        engine.state = DecisionState.NORMAL
        # Override meta-learner min_confidence to 0.0 so it doesn't interfere
        engine.meta.config.min_confidence = 0.0
        signal, conf = engine.should_override_signal("BUY", 0.7)
        assert signal == "BUY"
        assert conf == 0.7

    def test_below_meta_min_confidence_holds(self, engine):
        """Meta-learner min_confidence gate."""
        engine.state = DecisionState.NORMAL
        engine.meta.config.min_confidence = 0.8
        signal, conf = engine.should_override_signal("BUY", 0.5)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Trade result recording
# ---------------------------------------------------------------------------


class TestTradeRecording:
    def test_winning_trade_resets_consecutive_losses(self, engine):
        engine._consecutive_losses = 3
        engine.record_trade_result(
            pnl=50.0,
            strategy_signal="BUY",
            ml_signal="BUY",
            final_signal="BUY",
            strategy_confidence=0.8,
            ml_confidence=0.7,
            regime="trending_up",
        )
        assert engine._consecutive_losses == 0

    def test_losing_trade_increments_consecutive_losses(self, engine):
        engine.record_trade_result(
            pnl=-30.0,
            strategy_signal="BUY",
            ml_signal="BUY",
            final_signal="BUY",
            strategy_confidence=0.7,
            ml_confidence=0.6,
            regime="ranging",
        )
        assert engine._consecutive_losses == 1

    def test_daily_pnl_accumulates(self, engine):
        engine.record_trade_result(
            pnl=100.0,
            strategy_signal="BUY",
            ml_signal="BUY",
            final_signal="BUY",
            strategy_confidence=0.8,
            ml_confidence=0.7,
            regime="trending_up",
        )
        engine.record_trade_result(
            pnl=-40.0,
            strategy_signal="SELL",
            ml_signal="SELL",
            final_signal="SELL",
            strategy_confidence=0.7,
            ml_confidence=0.6,
            regime="ranging",
        )
        assert engine._daily_pnl == pytest.approx(60.0)

    def test_total_decisions_tracked(self, engine):
        assert engine._total_autonomous_decisions == 0
        engine.record_trade_result(
            pnl=10.0,
            strategy_signal="BUY",
            ml_signal="BUY",
            final_signal="BUY",
            strategy_confidence=0.8,
            ml_confidence=0.7,
            regime="ranging",
        )
        assert engine._total_autonomous_decisions == 1


# ---------------------------------------------------------------------------
# Orchestration instructions shape
# ---------------------------------------------------------------------------


class TestOrchestrate:
    def test_instructions_shape(self, engine):
        instructions = engine.orchestrate(cycle_count=1, current_capital=10000)
        assert "should_trade" in instructions
        assert "position_multiplier" in instructions
        assert "signal_weights" in instructions
        assert "autonomous_actions" in instructions
        assert isinstance(instructions["autonomous_actions"], list)

    def test_signal_weights_populated(self, engine):
        instructions = engine.orchestrate(cycle_count=1, current_capital=10000)
        weights = instructions["signal_weights"]
        assert weights is not None
        assert "strategy" in weights
        assert "ml" in weights


# ---------------------------------------------------------------------------
# Status and serialization
# ---------------------------------------------------------------------------


class TestStatusAndSerialization:
    def test_get_autonomous_status_shape(self, engine):
        status = engine.get_autonomous_status()
        assert status["state"] == "normal"
        assert "daily_pnl" in status
        assert "consecutive_losses" in status
        assert "healer" in status
        assert "recent_events" in status

    def test_to_dict_from_dict_round_trip(self, engine):
        engine._daily_pnl = -200
        engine._consecutive_losses = 3
        engine._total_autonomous_decisions = 42
        engine._last_evolution_cycle = 500
        engine.state = DecisionState.CAUTIOUS

        data = engine.to_dict()
        assert data["state"] == "cautious"
        assert data["consecutive_losses"] == 3

        engine2 = DecisionEngine(initial_capital=10000)
        engine2.from_dict(data)
        assert engine2.state == DecisionState.CAUTIOUS
        assert engine2._consecutive_losses == 3
        assert engine2._total_autonomous_decisions == 42
        assert engine2._last_evolution_cycle == 500

    def test_event_log_bounded(self, engine):
        """Event log is a deque with maxlen=1000."""
        for i in range(1100):
            engine._log_event("test", f"event {i}")
        assert len(engine.event_log) == 1000
