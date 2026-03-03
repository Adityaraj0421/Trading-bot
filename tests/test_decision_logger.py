# tests/test_decision_logger.py
"""Tests for DecisionLogger structured JSON logging."""

import json
from datetime import UTC, datetime, timedelta

from decision import ContextState, Decision, TriggerSignal
from decision_logger import DecisionLogger

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def make_context(tradeable: bool = True) -> ContextState:
    now = datetime.now(UTC)
    return ContextState(
        context_id="2026-03-03T14:15Z",
        swing_bias="bullish",
        allowed_directions=["long"] if tradeable else [],
        volatility_regime="normal",
        funding_pressure="neutral",
        whale_flow="accumulating",
        oi_trend="expanding_up",
        key_levels={"support": 88000.0, "resistance": 92000.0, "poc": 90000.0},
        risk_mode="normal",
        confidence=0.75,
        tradeable=tradeable,
        valid_until=now + timedelta(minutes=15),
        updated_at=now,
    )


def make_trigger(direction: str = "long") -> TriggerSignal:
    return TriggerSignal(
        trigger_id="test-uuid",
        source="momentum_1h",
        direction=direction,
        strength=0.70,
        urgency="normal",
        symbol_scope="BTC",
        reason="RSI crossed 50↑ + vol 1.8×",
        expires_at=datetime.now(UTC) + timedelta(minutes=75),
        raw_data={"rsi": 55.3, "macd": 12.5},
    )


def make_trade_decision() -> Decision:
    return Decision(
        action="trade",
        reason="ok",
        direction="long",
        route="spot",
        score=0.60,
    )


def make_reject_decision(reason: str = "context_not_tradeable") -> Decision:
    return Decision(action="reject", reason=reason)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestDecisionLogger:
    def test_log_does_not_raise_without_file(self):
        logger = DecisionLogger()  # no file
        context = make_context()
        logger.log(context, [make_trigger()], make_trade_decision())  # should not raise

    def test_recent_returns_empty_without_file(self):
        logger = DecisionLogger()
        assert logger.recent() == []

    def test_log_writes_jsonl_to_file(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)

        context = make_context()
        trigger = make_trigger()
        decision = make_trade_decision()
        logger.log(context, [trigger], decision)

        lines = log_file.read_text().splitlines()
        assert len(lines) == 1
        record = json.loads(lines[0])
        assert record["decision"]["action"] == "trade"

    def test_log_appends_multiple_records(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)

        for _ in range(3):
            logger.log(make_context(), [make_trigger()], make_trade_decision())

        lines = log_file.read_text().splitlines()
        assert len(lines) == 3

    def test_record_contains_context_fields(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)
        logger.log(make_context(), [make_trigger()], make_trade_decision())

        record = json.loads(log_file.read_text().splitlines()[0])
        ctx = record["context"]
        assert ctx["swing_bias"] == "bullish"
        assert ctx["confidence"] == 0.75
        assert ctx["tradeable"] is True
        assert "support" in ctx["key_levels"]

    def test_record_contains_trigger_fields(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)
        logger.log(make_context(), [make_trigger()], make_trade_decision())

        record = json.loads(log_file.read_text().splitlines()[0])
        assert len(record["triggers"]) == 1
        t = record["triggers"][0]
        assert t["source"] == "momentum_1h"
        assert t["direction"] == "long"
        assert t["strength"] == 0.70

    def test_record_contains_decision_fields(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)
        logger.log(make_context(), [make_trigger()], make_trade_decision())

        record = json.loads(log_file.read_text().splitlines()[0])
        d = record["decision"]
        assert d["action"] == "trade"
        assert d["route"] == "spot"
        assert d["score"] == 0.60

    def test_reject_decision_serialised_correctly(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)
        logger.log(
            make_context(tradeable=False),
            [],
            make_reject_decision("context_not_tradeable"),
        )

        record = json.loads(log_file.read_text().splitlines()[0])
        d = record["decision"]
        assert d["action"] == "reject"
        assert d["reason"] == "context_not_tradeable"
        assert d["score"] is None
        assert d["direction"] is None

    def test_recent_returns_last_n_records(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)

        for _ in range(10):
            logger.log(make_context(), [make_trigger()], make_trade_decision())

        recent = logger.recent(n=3)
        assert len(recent) == 3

    def test_recent_returns_all_when_fewer_than_n(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)

        logger.log(make_context(), [make_trigger()], make_trade_decision())
        logger.log(make_context(), [make_trigger()], make_trade_decision())

        recent = logger.recent(n=50)
        assert len(recent) == 2

    def test_record_has_timestamp(self, tmp_path):
        log_file = tmp_path / "decisions.jsonl"
        logger = DecisionLogger(log_path=log_file)
        logger.log(make_context(), [make_trigger()], make_trade_decision())

        record = json.loads(log_file.read_text().splitlines()[0])
        assert "ts" in record
        # Timestamp should be a valid ISO-8601 string
        datetime.fromisoformat(record["ts"])

    def test_creates_parent_directories(self, tmp_path):
        nested = tmp_path / "logs" / "phase9" / "decisions.jsonl"
        logger = DecisionLogger(log_path=nested)
        logger.log(make_context(), [], make_reject_decision())
        assert nested.exists()
