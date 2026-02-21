"""
Self-Healing Module
===================
Autonomous error recovery, circuit breaker pattern, and health monitoring.
Detects failures, attempts auto-recovery, and protects the system from
cascading errors.
"""

import logging
import time
import traceback
from enum import Enum
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
from typing import Optional, Callable

_log = logging.getLogger(__name__)


class ErrorSeverity(Enum):
    LOW = "low"           # Logging, display issues
    MEDIUM = "medium"     # Single component degraded
    HIGH = "high"         # Core component failing
    CRITICAL = "critical" # System integrity at risk


class CircuitState(Enum):
    CLOSED = "closed"       # Normal operation
    OPEN = "open"           # Failing — skip operations
    HALF_OPEN = "half_open" # Testing if recovered


@dataclass
class ErrorRecord:
    component: str
    error_type: str
    message: str
    severity: ErrorSeverity
    timestamp: datetime
    traceback_str: str = ""
    recovered: bool = False


@dataclass
class HealthMetrics:
    memory_ok: bool = True
    api_healthy: bool = True
    data_fresh: bool = True
    model_stale: bool = False
    error_rate: float = 0.0
    uptime_seconds: float = 0.0
    components_healthy: int = 0
    components_total: int = 0

    @property
    def overall_healthy(self) -> bool:
        critical = self.api_healthy and self.data_fresh and self.memory_ok
        return critical and self.error_rate < 0.3

    def to_dict(self) -> dict:
        return {
            "memory_ok": self.memory_ok,
            "api_healthy": self.api_healthy,
            "data_fresh": self.data_fresh,
            "model_stale": self.model_stale,
            "error_rate": round(self.error_rate, 4),
            "uptime_seconds": round(self.uptime_seconds, 1),
            "components_healthy": self.components_healthy,
            "components_total": self.components_total,
            "overall_healthy": self.overall_healthy,
        }


@dataclass
class CircuitBreaker:
    state: CircuitState = CircuitState.CLOSED
    failure_count: int = 0
    last_failure: Optional[datetime] = None
    last_success: Optional[datetime] = None
    open_since: Optional[datetime] = None
    half_open_attempts: int = 0
    failure_threshold: int = 3
    recovery_timeout: float = 60.0  # seconds before trying half-open
    max_half_open_attempts: int = 3


class SelfHealer:
    """
    Autonomous error recovery and health monitoring system.
    Uses circuit breaker pattern to prevent cascading failures.
    """

    COMPONENTS = [
        "data_fetcher", "model", "risk_manager", "executor",
        "regime_detector", "sentiment", "strategy_engine",
        "drift_detector", "state_manager", "logger",
    ]

    def __init__(self, max_errors: int = 200):
        self.start_time = datetime.now()
        self.error_history: deque = deque(maxlen=max_errors)
        self.circuit_breakers: dict[str, CircuitBreaker] = {
            comp: CircuitBreaker() for comp in self.COMPONENTS
        }
        self.recovery_actions: dict[str, list[Callable]] = {}
        self.recovery_attempts: dict[str, int] = defaultdict(int)
        self.last_health_check: Optional[datetime] = None
        self._cached_health: Optional[HealthMetrics] = None
        self._last_data_time: Optional[datetime] = None
        self._last_model_train: Optional[datetime] = None
        self._consecutive_data_failures: int = 0

    def record_error(self, component: str, error: Exception,
                     severity: ErrorSeverity = ErrorSeverity.MEDIUM):
        """Record an error and update circuit breaker state."""
        record = ErrorRecord(
            component=component,
            error_type=type(error).__name__,
            message=str(error)[:500],
            severity=severity,
            timestamp=datetime.now(),
            traceback_str=traceback.format_exc()[:1000],
        )
        self.error_history.append(record)

        # Update circuit breaker
        cb = self.circuit_breakers.get(component)
        if cb:
            cb.failure_count += 1
            cb.last_failure = datetime.now()

            if cb.state == CircuitState.CLOSED and cb.failure_count >= cb.failure_threshold:
                cb.state = CircuitState.OPEN
                cb.open_since = datetime.now()
                _log.warning("  [SelfHealer] CIRCUIT OPEN: %s (after %d failures)", component, cb.failure_count)

            elif cb.state == CircuitState.HALF_OPEN:
                cb.half_open_attempts += 1
                if cb.half_open_attempts >= cb.max_half_open_attempts:
                    cb.state = CircuitState.OPEN
                    cb.open_since = datetime.now()
                    _log.warning("  [SelfHealer] CIRCUIT RE-OPENED: %s", component)

        # Auto-recovery for critical errors
        if severity in (ErrorSeverity.HIGH, ErrorSeverity.CRITICAL):
            self.attempt_recovery(component)

    def record_success(self, component: str):
        """Record a successful operation — helps circuit breaker recover."""
        cb = self.circuit_breakers.get(component)
        if cb:
            cb.last_success = datetime.now()
            if cb.state == CircuitState.HALF_OPEN:
                cb.state = CircuitState.CLOSED
                cb.failure_count = 0
                cb.half_open_attempts = 0
                _log.info("  [SelfHealer] CIRCUIT CLOSED: %s recovered", component)
            elif cb.state == CircuitState.CLOSED:
                # Decay failure count on success
                cb.failure_count = max(0, cb.failure_count - 1)

    def get_circuit_state(self, component: str) -> CircuitState:
        """Get current circuit state, transitioning OPEN→HALF_OPEN if timeout elapsed."""
        cb = self.circuit_breakers.get(component)
        if not cb:
            return CircuitState.CLOSED

        if cb.state == CircuitState.OPEN and cb.open_since:
            elapsed = (datetime.now() - cb.open_since).total_seconds()
            # Exponential backoff: double timeout each time
            backoff = cb.recovery_timeout * (2 ** min(self.recovery_attempts[component], 5))
            if elapsed >= backoff:
                cb.state = CircuitState.HALF_OPEN
                cb.half_open_attempts = 0
                _log.info("  [SelfHealer] CIRCUIT HALF-OPEN: testing %s", component)

        return cb.state

    def is_component_available(self, component: str) -> bool:
        """Check if a component is available (circuit not OPEN)."""
        state = self.get_circuit_state(component)
        return state != CircuitState.OPEN

    def register_recovery_action(self, component: str, action: Callable):
        """Register a recovery action for a component."""
        if component not in self.recovery_actions:
            self.recovery_actions[component] = []
        self.recovery_actions[component].append(action)

    def attempt_recovery(self, component: str) -> bool:
        """Attempt to recover a failed component."""
        self.recovery_attempts[component] += 1
        attempt = self.recovery_attempts[component]
        _log.info("  [SelfHealer] Recovery attempt #%d for %s", attempt, component)

        actions = self.recovery_actions.get(component, [])
        for action in actions:
            try:
                action()
                self.record_success(component)
                _log.info("  [SelfHealer] Recovery SUCCESS: %s", component)
                return True
            except Exception as e:
                _log.error("  [SelfHealer] Recovery FAILED: %s - %s", component, e)

        return False

    def record_data_fetch(self, success: bool):
        """Track data fetcher health."""
        if success:
            self._last_data_time = datetime.now()
            self._consecutive_data_failures = 0
            self.record_success("data_fetcher")
        else:
            self._consecutive_data_failures += 1
            if self._consecutive_data_failures >= 3:
                self.record_error(
                    "data_fetcher",
                    RuntimeError(f"{self._consecutive_data_failures} consecutive data fetch failures"),
                    ErrorSeverity.HIGH,
                )

    def record_model_train(self):
        """Track model training time."""
        self._last_model_train = datetime.now()
        self.record_success("model")

    def check_health(self) -> HealthMetrics:
        """Comprehensive health check of all components."""
        now = datetime.now()
        metrics = HealthMetrics()

        # Uptime
        metrics.uptime_seconds = (now - self.start_time).total_seconds()

        # Data freshness (stale if no data in 10 minutes)
        if self._last_data_time:
            data_age = (now - self._last_data_time).total_seconds()
            metrics.data_fresh = data_age < 600
        else:
            metrics.data_fresh = True  # No data yet, not stale

        # Model staleness (stale if not trained in 24h)
        if self._last_model_train:
            model_age = (now - self._last_model_train).total_seconds()
            metrics.model_stale = model_age > 86400
        else:
            metrics.model_stale = False

        # API health (based on data_fetcher circuit)
        metrics.api_healthy = self.is_component_available("data_fetcher")

        # Error rate (errors per minute over actual window)
        window_minutes = 10
        cutoff = now - timedelta(minutes=window_minutes)
        recent_errors = sum(1 for e in self.error_history if e.timestamp > cutoff)
        metrics.error_rate = recent_errors / max(window_minutes, 1)  # per minute

        # Component health count
        metrics.components_total = len(self.COMPONENTS)
        metrics.components_healthy = sum(
            1 for c in self.COMPONENTS if self.is_component_available(c)
        )

        # Memory check — use psutil for real memory pressure detection
        try:
            import psutil
            mem = psutil.virtual_memory()
            # Flag if less than 10% memory available
            metrics.memory_ok = mem.percent < 90
        except ImportError:
            # Fallback: check if process RSS is reasonable (< 2GB)
            try:
                import resource
                usage_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024)
                metrics.memory_ok = usage_mb < 2048
            except (ImportError, AttributeError):
                metrics.memory_ok = True  # Can't check, assume OK

        self._cached_health = metrics
        self.last_health_check = now
        return metrics

    def get_error_summary(self) -> dict:
        """Get summary of recent errors by component."""
        summary = defaultdict(lambda: {"count": 0, "last": None, "severity": "low"})
        cutoff = datetime.now() - timedelta(hours=1)

        for err in self.error_history:
            if err.timestamp > cutoff:
                comp = summary[err.component]
                comp["count"] += 1
                comp["last"] = err.timestamp.isoformat()
                # Keep highest severity
                sev_order = {"low": 0, "medium": 1, "high": 2, "critical": 3}
                if sev_order.get(err.severity.value, 0) > sev_order.get(comp["severity"], 0):
                    comp["severity"] = err.severity.value

        return dict(summary)

    def get_status(self) -> dict:
        """Full self-healer status report."""
        health = self._cached_health or self.check_health()
        return {
            "health": health.to_dict(),
            "circuits": {
                comp: {
                    "state": cb.state.value,
                    "failures": cb.failure_count,
                    "recovery_attempts": self.recovery_attempts[comp],
                }
                for comp, cb in self.circuit_breakers.items()
                if cb.failure_count > 0 or cb.state != CircuitState.CLOSED
            },
            "total_errors": len(self.error_history),
            "error_summary": self.get_error_summary(),
        }

    def to_dict(self) -> dict:
        """Serialize for state persistence."""
        return {
            "recovery_attempts": dict(self.recovery_attempts),
            "total_errors": len(self.error_history),
            "start_time": self.start_time.isoformat(),
        }

    def from_dict(self, data: dict):
        """Restore from state."""
        self.recovery_attempts = defaultdict(int, data.get("recovery_attempts", {}))
