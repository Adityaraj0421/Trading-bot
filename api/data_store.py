"""
DataStore — Thread-Safe Agent↔API Bridge (v7.0)
==========================================
The agent writes cycle snapshots, equity points, trades, events,
notifications, and v7.0 module status.
The API reads them. All operations are thread-safe via a single lock.
"""

from __future__ import annotations

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

    def __init__(self, max_history_size: int = 10000) -> None:
        """Initialize the DataStore with empty in-memory collections.

        All mutable state is guarded by ``self._lock`` (a ``threading.Lock``).
        Writers (agent thread) and readers (API request handlers) must acquire
        this lock for every access to maintain consistency.

        Args:
            max_history_size: Maximum number of equity data points to retain
                in the circular deque (default 10 000).
        """
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
        """Set the WebSocket broadcast callback for real-time push to clients.

        Called once from ``api/server.py`` during FastAPI lifespan startup,
        wiring ``WebSocketManager.broadcast_sync`` as the callback. After this
        every writer method (``update_snapshot``, ``append_trade``, etc.) will
        also push its data to all connected dashboard clients.

        Args:
            callback: Callable with signature ``(event_type: str, data: dict)``
                that schedules an async WebSocket broadcast on the FastAPI loop.
        """
        self._broadcast = callback

    def set_decision_engine(self, engine: Any) -> None:
        """Store a reference to the DecisionEngine for API-triggered controls.

        Args:
            engine: The live ``DecisionEngine`` instance used by the agent.
        """
        self._decision_engine = engine

    def get_decision_engine(self) -> Any:
        """Return the stored DecisionEngine reference.

        Used by API routes (``/autonomous/halt``, ``/autonomous/resume``,
        ``/autonomous/force-close``) and Telegram command handlers to invoke
        kill-switch and override operations.

        Returns:
            The ``DecisionEngine`` instance, or ``None`` if the agent has not
            started yet.
        """
        return self._decision_engine

    # --- Writer methods (called by agent) ---

    def update_snapshot(self, snapshot: dict) -> None:
        """Replace the current agent cycle snapshot and push to WebSocket clients.

        Stamps the snapshot with ``updated_at`` (ISO 8601 timestamp) before
        storing. The /health endpoint uses this timestamp to detect a stale
        agent (>300 s since last update → 503 degraded).

        Args:
            snapshot: Dict produced by the agent each cycle containing
                status, equity, positions, PnL, regime, etc.
        """
        with self._lock:
            self._snapshot = snapshot.copy()
            self._snapshot["updated_at"] = datetime.now().isoformat()
            data = self._snapshot.copy()
        if self._broadcast:
            self._broadcast("snapshot", data)

    def append_equity(self, equity: float, timestamp: str) -> None:
        """Append an equity data point and broadcast to WebSocket clients.

        Args:
            equity: Current total portfolio value in quote currency.
            timestamp: ISO 8601 timestamp string for this data point.
        """
        point = {"equity": equity, "timestamp": timestamp}
        with self._lock:
            self._equity_history.append(point)
        if self._broadcast:
            self._broadcast("equity", point)

    def append_trade(self, trade: dict) -> None:
        """Append a trade record to the in-memory log and broadcast to clients.

        This is the session-only trade log. For persistent storage across
        restarts, see ``TradeDB`` (SQLite). API routes prefer ``TradeDB``
        when available and fall back to this log.

        Args:
            trade: Trade dict with keys such as ``symbol``, ``side``,
                ``entry_price``, ``exit_price``, ``pnl``, ``strategy``.
        """
        with self._lock:
            self._trade_log.append(trade)
        if self._broadcast:
            self._broadcast("trade", trade)

    def append_event(self, event: dict) -> None:
        """Append an autonomous-mode decision event and broadcast to clients.

        The event deque holds up to 1 000 entries. Events are session-only
        (not persisted to SQLite); only the ``total_autonomous_decisions``
        counter survives restarts via agent state.

        Args:
            event: Event dict from ``decision_engine.event_log``.
        """
        with self._lock:
            self._events.append(event)
        if self._broadcast:
            self._broadcast("event", event)

    def update_intelligence(self, signals: dict) -> None:
        """Replace the latest intelligence signals snapshot.

        Args:
            signals: Aggregated signals dict from the intelligence layer.
        """
        with self._lock:
            self._intelligence = signals.copy()

    def update_arbitrage(self, data: dict) -> None:
        """Replace the latest arbitrage opportunities snapshot.

        Args:
            data: Dict of detected cross-exchange arbitrage opportunities.
        """
        with self._lock:
            self._arbitrage = data.copy()

    def update_backtest_results(self, results: list[dict]) -> None:
        """Replace the stored backtest results list.

        Args:
            results: List of backtest result dicts, typically appended
                incrementally by the background backtest thread.
        """
        with self._lock:
            self._backtest_results = results

    def update_monte_carlo(self, results: dict) -> None:
        """Replace the stored Monte Carlo simulation results.

        Args:
            results: Dict with ``monte_carlo``, ``stress_tests``, and
                ``status`` keys produced by the risk simulation background thread.
        """
        with self._lock:
            self._monte_carlo = results.copy()

    # --- Reader methods (called by API) ---

    def get_snapshot(self) -> dict:
        """Return a thread-safe copy of the current agent cycle snapshot.

        Returns:
            Dict with keys including ``status``, ``equity``, ``positions``,
            ``total_pnl``, ``win_rate``, ``regime``, ``cycle``,
            ``trading_mode``, ``autonomous``, and ``updated_at``.
            Returns an empty dict if no snapshot has been written yet.
        """
        with self._lock:
            return self._snapshot.copy()

    def get_equity_history(self) -> list[dict]:
        """Return the full in-memory equity history as a list.

        Returns:
            List of dicts with ``equity`` (float) and ``timestamp`` (str)
            keys, in chronological order.
        """
        with self._lock:
            return list(self._equity_history)

    def get_trade_log(self) -> list[dict]:
        """Return a shallow copy of the in-memory session trade log.

        Returns:
            List of trade dicts appended by ``append_trade()``. Session-only —
            does not survive server restarts. Prefer ``TradeDB`` when available.
        """
        with self._lock:
            return self._trade_log.copy()

    def get_events(self, limit: int = 100) -> list[dict]:
        """Return the most recent autonomous-mode events.

        Args:
            limit: Maximum number of most-recent events to return.

        Returns:
            List of event dicts (newest last), at most *limit* entries.
        """
        with self._lock:
            events = list(self._events)
            return events[-limit:]

    def get_intelligence(self) -> dict:
        """Return a thread-safe copy of the latest intelligence signals.

        Returns:
            Aggregated signals dict, or empty dict if not yet populated.
        """
        with self._lock:
            return self._intelligence.copy()

    def get_arbitrage(self) -> dict:
        """Return a thread-safe copy of the latest arbitrage opportunities.

        Returns:
            Arbitrage opportunities dict, or empty dict if not yet populated.
        """
        with self._lock:
            return self._arbitrage.copy()

    def get_backtest_results(self) -> list[dict]:
        """Return a shallow copy of the stored backtest results.

        Returns:
            List of backtest result dicts, or empty list if none have run.
        """
        with self._lock:
            return self._backtest_results.copy()

    def get_monte_carlo(self) -> dict:
        """Return a thread-safe copy of the Monte Carlo simulation results.

        Returns:
            Dict with ``monte_carlo``, ``stress_tests``, and ``status`` keys,
            or empty dict if no simulation has been run.
        """
        with self._lock:
            return self._monte_carlo.copy()

    # --- v7.0 writer methods ---

    def append_notification(self, notification: dict) -> None:
        """Log a notification event for the /notifications API endpoint.

        Called by ``notifier.py:_send_all()`` — the single funnel for all
        outbound alerts. Automatically stamps with the current ISO timestamp
        if ``notification`` does not already contain one.

        Args:
            notification: Dict with at minimum ``level`` and ``message`` keys.
        """
        with self._lock:
            notification.setdefault("timestamp", datetime.now().isoformat())
            self._notifications.append(notification)

    def update_system_modules(self, modules: dict) -> None:
        """Update the v7.0 system module status and stamp with current time.

        Args:
            modules: Dict mapping module name to its status dict (e.g.
                ``{"websocket": {"enabled": True, "status": "running"}}``).
        """
        with self._lock:
            self._system_modules = modules.copy()
            self._system_modules["updated_at"] = datetime.now().isoformat()

    def update_rate_limiter_stats(self, stats: dict) -> None:
        """Replace the stored rate limiter usage statistics.

        Args:
            stats: Dict with ``requests_used``, ``orders_used``, and
                configured limit keys.
        """
        with self._lock:
            self._rate_limiter_stats = stats.copy()

    def set_trade_db(self, trade_db: Any) -> None:
        """Store a reference to the TradeDB instance for persistent queries.

        Args:
            trade_db: The ``TradeDB`` SQLite wrapper instance.
        """
        self._trade_db = trade_db

    # --- v7.0 reader methods ---

    def get_notifications(self, limit: int = 100) -> list[dict]:
        """Return the most recent notification log entries.

        Args:
            limit: Maximum number of most-recent entries to return.

        Returns:
            List of notification dicts (newest last), at most *limit* entries.
        """
        with self._lock:
            notifs = list(self._notifications)
            return notifs[-limit:]

    def get_system_modules(self) -> dict:
        """Return a thread-safe copy of the v7.0 system module statuses.

        Returns:
            Dict of module status dicts, or empty dict if not yet populated.
        """
        with self._lock:
            return self._system_modules.copy()

    def get_rate_limiter_stats(self) -> dict:
        """Return a thread-safe copy of the rate limiter usage statistics.

        Returns:
            Dict with request/order usage counts, or empty dict if not set.
        """
        with self._lock:
            return self._rate_limiter_stats.copy()

    def get_trade_db(self) -> Any:
        """Return the stored TradeDB reference for persistent analytics queries.

        Returns:
            The ``TradeDB`` instance, or ``None`` if not yet wired
            (e.g. ``ENABLE_TRADE_DB=false`` or agent not yet started).
        """
        return self._trade_db
