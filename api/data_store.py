"""
DataStore — Thread-Safe Agent↔API Bridge (v7.0)
==========================================
The agent writes cycle snapshots, equity points, trades, events,
notifications, and v7.0 module status.
The API reads them. All operations are thread-safe via a single lock.
"""

import threading
from collections import deque
from collections.abc import Callable
from datetime import datetime
from typing import Any


class DataStore:
    """
    Thread-safe in-memory data bridge between the trading agent and API.
    The agent calls update_snapshot(), append_equity(), append_trade(),
    and append_event() from its main loop. The API reads via get_*() methods.
    v7.0 adds: notifications log, system module status, trade DB reference.
    """

    def __init__(self, max_history_size: int = 10000):
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._equity_history: deque = deque(maxlen=max_history_size)
        self._trade_log: list[dict] = []
        self._events: deque = deque(maxlen=1000)
        self._intelligence: dict = {}
        self._arbitrage: dict = {}
        self._backtest_results: list[dict] = []
        self._monte_carlo: dict = {}
        self._decision_engine = None  # Reference for kill switch / alerts

        # v7.0 additions
        self._notifications: deque = deque(maxlen=500)
        self._system_modules: dict = {}
        self._rate_limiter_stats: dict = {}
        self._trade_db = None  # Reference to TradeDB instance

        # v7.1: WebSocket broadcast callback (set by server.py)
        self._broadcast: Callable[[str, dict], None] | None = None

    def set_broadcast_callback(self, callback: Callable[[str, dict], None]) -> None:
        """Set a callback for broadcasting state changes to WebSocket clients."""
        self._broadcast = callback

    def set_decision_engine(self, engine: Any) -> None:
        """Store reference to DecisionEngine for production safeguards."""
        self._decision_engine = engine

    def get_decision_engine(self) -> Any:
        """Get DecisionEngine reference (for kill switch, alerts, manual override)."""
        return self._decision_engine

    # --- Writer methods (called by agent) ---

    def update_snapshot(self, snapshot: dict) -> None:
        """Replace the current agent snapshot and broadcast to WebSocket clients."""
        with self._lock:
            self._snapshot = snapshot.copy()
            self._snapshot["updated_at"] = datetime.now().isoformat()
            data = self._snapshot.copy()
        if self._broadcast:
            self._broadcast("snapshot", data)

    def append_equity(self, equity: float, timestamp: str) -> None:
        """Append an equity data point and broadcast to WebSocket clients."""
        point = {"equity": equity, "timestamp": timestamp}
        with self._lock:
            self._equity_history.append(point)
        if self._broadcast:
            self._broadcast("equity", point)

    def append_trade(self, trade: dict) -> None:
        """Append a trade record and broadcast to WebSocket clients."""
        with self._lock:
            self._trade_log.append(trade)
        if self._broadcast:
            self._broadcast("trade", trade)

    def append_event(self, event: dict) -> None:
        """Append an autonomous-mode event and broadcast to WebSocket clients."""
        with self._lock:
            self._events.append(event)
        if self._broadcast:
            self._broadcast("event", event)

    def update_intelligence(self, signals: dict) -> None:
        """Replace the latest intelligence signals snapshot."""
        with self._lock:
            self._intelligence = signals.copy()

    def update_arbitrage(self, data: dict) -> None:
        """Replace the latest arbitrage opportunities snapshot."""
        with self._lock:
            self._arbitrage = data.copy()

    def update_backtest_results(self, results: list[dict]) -> None:
        """Replace the stored backtest results list."""
        with self._lock:
            self._backtest_results = results

    def update_monte_carlo(self, results: dict) -> None:
        """Replace the stored Monte Carlo simulation results."""
        with self._lock:
            self._monte_carlo = results.copy()

    # --- Reader methods (called by API) ---

    def get_snapshot(self) -> dict:
        """Return a copy of the current agent snapshot."""
        with self._lock:
            return self._snapshot.copy()

    def get_equity_history(self) -> list[dict]:
        """Return the full equity history as a list."""
        with self._lock:
            return list(self._equity_history)

    def get_trade_log(self) -> list[dict]:
        """Return a copy of the full trade log."""
        with self._lock:
            return self._trade_log.copy()

    def get_events(self, limit: int = 100) -> list[dict]:
        """Return the most recent *limit* autonomous-mode events."""
        with self._lock:
            events = list(self._events)
            return events[-limit:]

    def get_intelligence(self) -> dict:
        """Return a copy of the latest intelligence signals."""
        with self._lock:
            return self._intelligence.copy()

    def get_arbitrage(self) -> dict:
        """Return a copy of the latest arbitrage opportunities."""
        with self._lock:
            return self._arbitrage.copy()

    def get_backtest_results(self) -> list[dict]:
        """Return a copy of the stored backtest results."""
        with self._lock:
            return self._backtest_results.copy()

    def get_monte_carlo(self) -> dict:
        """Return a copy of the Monte Carlo simulation results."""
        with self._lock:
            return self._monte_carlo.copy()

    # --- v7.0 writer methods ---

    def append_notification(self, notification: dict) -> None:
        """Log a notification event (Telegram/Discord/Email)."""
        with self._lock:
            notification.setdefault("timestamp", datetime.now().isoformat())
            self._notifications.append(notification)

    def update_system_modules(self, modules: dict) -> None:
        """Update the status of v7.0 system modules."""
        with self._lock:
            self._system_modules = modules.copy()
            self._system_modules["updated_at"] = datetime.now().isoformat()

    def update_rate_limiter_stats(self, stats: dict) -> None:
        """Replace the stored rate limiter usage statistics."""
        with self._lock:
            self._rate_limiter_stats = stats.copy()

    def set_trade_db(self, trade_db: Any) -> None:
        """Store reference to TradeDB for query routes."""
        self._trade_db = trade_db

    # --- v7.0 reader methods ---

    def get_notifications(self, limit: int = 100) -> list[dict]:
        """Return the most recent *limit* notification log entries."""
        with self._lock:
            notifs = list(self._notifications)
            return notifs[-limit:]

    def get_system_modules(self) -> dict:
        """Return a copy of the v7.0 system module statuses."""
        with self._lock:
            return self._system_modules.copy()

    def get_rate_limiter_stats(self) -> dict:
        """Return a copy of the rate limiter usage statistics."""
        with self._lock:
            return self._rate_limiter_stats.copy()

    def get_trade_db(self) -> Any:
        """Get TradeDB reference (for analytics queries)."""
        return self._trade_db
