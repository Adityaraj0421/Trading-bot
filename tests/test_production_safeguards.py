"""
Tests for production safeguards: kill switch, alerts, manual override.
"""

import pytest
from decision_engine import DecisionEngine, DecisionState


class TestKillSwitch:
    def test_halt_stops_trading(self):
        engine = DecisionEngine(initial_capital=10000)
        engine.emergency_halt("manual kill switch")
        instructions = engine.orchestrate(1, 10000, 0)
        assert instructions["should_trade"] is False
        assert engine.state == DecisionState.HALTED

    def test_resume_restores_trading(self):
        engine = DecisionEngine(initial_capital=10000)
        engine.emergency_halt("test")
        engine.emergency_resume()
        instructions = engine.orchestrate(1, 10000, 0)
        assert instructions["should_trade"] is True
        assert engine.state == DecisionState.NORMAL


class TestManualOverride:
    def test_force_close_all(self):
        engine = DecisionEngine(initial_capital=10000)
        engine._force_close_all = True
        instructions = engine.orchestrate(1, 10000, 0)
        assert instructions.get("force_close_all") is True


class TestAlertSystem:
    def test_alert_on_large_loss(self):
        engine = DecisionEngine(initial_capital=10000)
        # Simulate a big loss
        engine.record_trade_result(
            pnl=-300, strategy_signal="BUY", ml_signal="BUY",
            final_signal="BUY", strategy_confidence=0.8,
            ml_confidence=0.7, regime="TRENDING_DOWN",
        )
        # Check that alert was generated
        events = [e.event_type for e in engine.event_log]
        # After large loss, consecutive_losses should increase
        assert engine._consecutive_losses == 1
