"""
Unit tests for self_healer.py — CircuitBreaker, HealthMetrics, SelfHealer.

Tests circuit breaker state machine (CLOSED→OPEN→HALF_OPEN→CLOSED),
error recording, recovery actions, data/model freshness, and health metrics.
"""

import pytest
from datetime import datetime, timedelta
from self_healer import (
    ErrorSeverity, CircuitState, CircuitBreaker,
    HealthMetrics, ErrorRecord, SelfHealer,
)


# ---------------------------------------------------------------------------
# HealthMetrics
# ---------------------------------------------------------------------------

class TestHealthMetrics:
    def test_overall_healthy_when_all_good(self):
        m = HealthMetrics(api_healthy=True, data_fresh=True, memory_ok=True, error_rate=0.1)
        assert m.overall_healthy is True

    def test_overall_unhealthy_when_api_down(self):
        m = HealthMetrics(api_healthy=False, data_fresh=True, memory_ok=True, error_rate=0.0)
        assert m.overall_healthy is False

    def test_overall_unhealthy_when_stale_data(self):
        m = HealthMetrics(api_healthy=True, data_fresh=False, memory_ok=True, error_rate=0.0)
        assert m.overall_healthy is False

    def test_overall_unhealthy_when_high_error_rate(self):
        m = HealthMetrics(api_healthy=True, data_fresh=True, memory_ok=True, error_rate=0.5)
        assert m.overall_healthy is False

    def test_to_dict_shape(self):
        m = HealthMetrics(components_healthy=8, components_total=10)
        d = m.to_dict()
        assert d["components_healthy"] == 8
        assert d["components_total"] == 10
        assert "overall_healthy" in d
        assert "error_rate" in d


# ---------------------------------------------------------------------------
# CircuitBreaker defaults
# ---------------------------------------------------------------------------

class TestCircuitBreaker:
    def test_default_state_is_closed(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0

    def test_threshold_default(self):
        cb = CircuitBreaker()
        assert cb.failure_threshold == 3
        assert cb.recovery_timeout == 60.0


# ---------------------------------------------------------------------------
# SelfHealer
# ---------------------------------------------------------------------------

@pytest.fixture()
def healer():
    return SelfHealer(max_errors=100)


class TestRecordError:
    def test_error_recorded(self, healer):
        healer.record_error("model", ValueError("bad input"), ErrorSeverity.LOW)
        assert len(healer.error_history) == 1
        assert healer.error_history[0].component == "model"

    def test_circuit_opens_after_threshold(self, healer):
        for _ in range(3):
            healer.record_error("executor", RuntimeError("fail"), ErrorSeverity.MEDIUM)
        cb = healer.circuit_breakers["executor"]
        assert cb.state == CircuitState.OPEN

    def test_circuit_stays_closed_below_threshold(self, healer):
        healer.record_error("executor", RuntimeError("fail"), ErrorSeverity.LOW)
        healer.record_error("executor", RuntimeError("fail"), ErrorSeverity.LOW)
        cb = healer.circuit_breakers["executor"]
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 2

    def test_error_history_bounded(self, healer):
        for i in range(150):
            healer.record_error("model", RuntimeError(f"err {i}"), ErrorSeverity.LOW)
        assert len(healer.error_history) <= 100

    def test_high_severity_triggers_recovery(self, healer):
        """HIGH/CRITICAL errors automatically attempt recovery."""
        recovered = []
        healer.register_recovery_action("model", lambda: recovered.append(True))
        healer.record_error("model", RuntimeError("crash"), ErrorSeverity.HIGH)
        assert len(recovered) == 1  # Recovery action was called

    def test_half_open_failure_reopens_circuit(self, healer):
        """Failing during HALF_OPEN sends circuit back to OPEN."""
        cb = healer.circuit_breakers["executor"]
        cb.state = CircuitState.HALF_OPEN
        cb.max_half_open_attempts = 1  # Fail after 1 attempt
        healer.record_error("executor", RuntimeError("still broken"), ErrorSeverity.MEDIUM)
        assert cb.state == CircuitState.OPEN


class TestRecordSuccess:
    def test_success_decays_failure_count(self, healer):
        cb = healer.circuit_breakers["model"]
        cb.failure_count = 2
        healer.record_success("model")
        assert cb.failure_count == 1

    def test_success_closes_half_open_circuit(self, healer):
        cb = healer.circuit_breakers["executor"]
        cb.state = CircuitState.HALF_OPEN
        cb.failure_count = 5
        healer.record_success("executor")
        assert cb.state == CircuitState.CLOSED
        assert cb.failure_count == 0
        assert cb.half_open_attempts == 0

    def test_success_on_closed_never_goes_negative(self, healer):
        cb = healer.circuit_breakers["model"]
        cb.failure_count = 0
        healer.record_success("model")
        assert cb.failure_count == 0


class TestCircuitStateTransitions:
    def test_open_transitions_to_half_open_after_timeout(self, healer):
        cb = healer.circuit_breakers["data_fetcher"]
        cb.state = CircuitState.OPEN
        # Set open_since far enough in the past to exceed recovery_timeout
        cb.open_since = datetime.now() - timedelta(seconds=120)
        state = healer.get_circuit_state("data_fetcher")
        assert state == CircuitState.HALF_OPEN

    def test_open_stays_open_before_timeout(self, healer):
        cb = healer.circuit_breakers["data_fetcher"]
        cb.state = CircuitState.OPEN
        cb.open_since = datetime.now()  # Just opened
        state = healer.get_circuit_state("data_fetcher")
        assert state == CircuitState.OPEN

    def test_exponential_backoff_on_repeated_recovery(self, healer):
        """After multiple recovery attempts, timeout doubles each time."""
        healer.recovery_attempts["data_fetcher"] = 3  # 2^3 = 8x multiplier
        cb = healer.circuit_breakers["data_fetcher"]
        cb.state = CircuitState.OPEN
        cb.recovery_timeout = 60.0
        # Set open_since to 2 minutes ago (120s) — but backoff is 60 * 8 = 480s
        cb.open_since = datetime.now() - timedelta(seconds=120)
        state = healer.get_circuit_state("data_fetcher")
        assert state == CircuitState.OPEN  # Still open (120 < 480)

    def test_unknown_component_returns_closed(self, healer):
        state = healer.get_circuit_state("nonexistent_component")
        assert state == CircuitState.CLOSED

    def test_is_component_available_closed(self, healer):
        assert healer.is_component_available("model") is True

    def test_is_component_available_open(self, healer):
        cb = healer.circuit_breakers["model"]
        cb.state = CircuitState.OPEN
        cb.open_since = datetime.now()
        assert healer.is_component_available("model") is False


class TestRecoveryActions:
    def test_recovery_increments_attempt_count(self, healer):
        healer.attempt_recovery("model")
        assert healer.recovery_attempts["model"] == 1

    def test_successful_recovery_returns_true(self, healer):
        healer.register_recovery_action("model", lambda: None)
        result = healer.attempt_recovery("model")
        assert result is True

    def test_failed_recovery_returns_false(self, healer):
        def fail():
            raise RuntimeError("recovery failed")
        healer.register_recovery_action("model", fail)
        result = healer.attempt_recovery("model")
        assert result is False

    def test_no_actions_returns_false(self, healer):
        result = healer.attempt_recovery("model")
        assert result is False

    def test_register_multiple_actions(self, healer):
        calls = []
        def action_a():
            raise RuntimeError("a fails")
        def action_b():
            calls.append("b")
        healer.register_recovery_action("model", action_a)
        healer.register_recovery_action("model", action_b)
        result = healer.attempt_recovery("model")
        # action_a fails, action_b succeeds
        assert result is True
        assert "b" in calls


class TestDataFetchTracking:
    def test_successful_fetch_resets_counter(self, healer):
        healer._consecutive_data_failures = 5
        healer.record_data_fetch(success=True)
        assert healer._consecutive_data_failures == 0

    def test_failed_fetch_increments_counter(self, healer):
        healer.record_data_fetch(success=False)
        assert healer._consecutive_data_failures == 1

    def test_three_failures_records_high_error(self, healer):
        for _ in range(3):
            healer.record_data_fetch(success=False)
        assert healer._consecutive_data_failures == 3
        # Should have recorded an error
        assert any(e.component == "data_fetcher" for e in healer.error_history)

    def test_model_train_updates_timestamp(self, healer):
        healer.record_model_train()
        assert healer._last_model_train is not None


class TestHealthCheck:
    def test_returns_health_metrics(self, healer):
        metrics = healer.check_health()
        assert isinstance(metrics, HealthMetrics)
        assert metrics.uptime_seconds >= 0
        assert metrics.memory_ok is True

    def test_all_components_healthy_initially(self, healer):
        metrics = healer.check_health()
        assert metrics.components_healthy == metrics.components_total

    def test_data_fresh_when_recent(self, healer):
        healer._last_data_time = datetime.now()
        metrics = healer.check_health()
        assert metrics.data_fresh is True

    def test_data_stale_when_old(self, healer):
        healer._last_data_time = datetime.now() - timedelta(minutes=15)
        metrics = healer.check_health()
        assert metrics.data_fresh is False

    def test_model_not_stale_initially(self, healer):
        metrics = healer.check_health()
        assert metrics.model_stale is False

    def test_model_stale_after_24h(self, healer):
        healer._last_model_train = datetime.now() - timedelta(hours=25)
        metrics = healer.check_health()
        assert metrics.model_stale is True

    def test_error_rate_calculation(self, healer):
        for _ in range(10):
            healer.record_error("model", RuntimeError("err"), ErrorSeverity.LOW)
        metrics = healer.check_health()
        assert metrics.error_rate == pytest.approx(1.0)  # 10 errors / 10 min

    def test_open_circuit_affects_api_health(self, healer):
        cb = healer.circuit_breakers["data_fetcher"]
        cb.state = CircuitState.OPEN
        cb.open_since = datetime.now()  # Just opened, won't transition
        metrics = healer.check_health()
        assert metrics.api_healthy is False

    def test_caches_health(self, healer):
        healer.check_health()
        assert healer._cached_health is not None
        assert healer.last_health_check is not None


class TestErrorSummary:
    def test_empty_summary(self, healer):
        summary = healer.get_error_summary()
        assert summary == {}

    def test_summary_groups_by_component(self, healer):
        healer.record_error("model", RuntimeError("a"), ErrorSeverity.LOW)
        healer.record_error("model", RuntimeError("b"), ErrorSeverity.HIGH)
        healer.record_error("executor", RuntimeError("c"), ErrorSeverity.MEDIUM)
        summary = healer.get_error_summary()
        assert summary["model"]["count"] == 2
        assert summary["model"]["severity"] == "high"  # Highest severity wins
        assert summary["executor"]["count"] == 1


class TestSerialization:
    def test_get_status_shape(self, healer):
        status = healer.get_status()
        assert "health" in status
        assert "circuits" in status
        assert "total_errors" in status
        assert "error_summary" in status

    def test_to_dict_round_trip(self, healer):
        healer.attempt_recovery("model")
        healer.attempt_recovery("executor")
        data = healer.to_dict()
        assert data["recovery_attempts"]["model"] == 1

        healer2 = SelfHealer()
        healer2.from_dict(data)
        assert healer2.recovery_attempts["model"] == 1
        assert healer2.recovery_attempts["executor"] == 1

    def test_circuits_only_shows_non_default(self, healer):
        """get_status circuits dict only includes components with failures."""
        healer.record_error("model", RuntimeError("err"), ErrorSeverity.LOW)
        status = healer.get_status()
        assert "model" in status["circuits"]
        assert "executor" not in status["circuits"]  # No failures
