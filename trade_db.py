"""
Trade Database — SQLite persistence for trade history and performance tracking.
================================================================
Replaces JSON-based trade history with a proper database for:
  - Full trade history with queryable fields
  - Performance metrics by strategy, regime, time period
  - Equity curve persistence
  - System event logging
  - Daily summary generation

Uses SQLite (zero config, file-based) with WAL mode for concurrent reads.
"""

from __future__ import annotations

import json
import logging
import os
import sqlite3
from collections.abc import Generator
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from config import Config

_log = logging.getLogger(__name__)


class TradeDB:
    """SQLite-backed trade and performance database."""

    def __init__(self, db_path: str | None = None) -> None:
        """Initialise the trade database, creating tables if they do not exist.

        Args:
            db_path: Absolute path to the SQLite database file.  Defaults to
                ``<Config.DATA_DIR>/trades.db`` when not provided.
        """
        self.db_path = db_path or os.path.join(getattr(Config, "DATA_DIR", "."), "trades.db")
        self._init_db()

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        """Context manager that yields a WAL-mode SQLite connection.

        Commits on success; rolls back and re-raises on any exception.

        Yields:
            An open :class:`sqlite3.Connection` with ``row_factory`` set to
            :class:`sqlite3.Row` for dict-like row access.

        Raises:
            Exception: Any exception raised inside the with block is rolled
                back and re-raised unchanged.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")  # Better concurrent access
        conn.execute("PRAGMA foreign_keys=ON")
        try:
            yield conn
            conn.commit()
        except Exception as e:
            conn.rollback()
            _log.error("Transaction failed, rolling back: %s", e, exc_info=True)
            raise
        finally:
            conn.close()

    def close(self) -> None:
        """Close the database.

        This is a no-op because connections are opened and closed per query.
        Provided for API symmetry so callers can call ``db.close()`` without
        checking the implementation.
        """
        _log.info("TradeDB closed: %s", self.db_path)

    def _init_db(self) -> None:
        """Create all required tables and indexes if they do not already exist."""
        with self._conn() as conn:
            conn.executescript("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol TEXT NOT NULL,
                    side TEXT NOT NULL,
                    entry_price REAL NOT NULL,
                    exit_price REAL,
                    quantity REAL NOT NULL,
                    pnl_gross REAL DEFAULT 0,
                    pnl_net REAL DEFAULT 0,
                    fees_paid REAL DEFAULT 0,
                    slippage_cost REAL DEFAULT 0,
                    entry_time TEXT NOT NULL,
                    exit_time TEXT,
                    exit_reason TEXT,
                    strategy_name TEXT,
                    regime TEXT,
                    hold_bars INTEGER DEFAULT 0,
                    confidence REAL DEFAULT 0,
                    stop_loss REAL,
                    take_profit REAL,
                    trailing_stop REAL,
                    status TEXT DEFAULT 'open',
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                );

                CREATE TABLE IF NOT EXISTS equity_snapshots (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    equity REAL NOT NULL,
                    capital REAL NOT NULL,
                    unrealized_pnl REAL DEFAULT 0,
                    open_positions INTEGER DEFAULT 0,
                    cycle INTEGER DEFAULT 0
                );

                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    severity TEXT DEFAULT 'info',
                    description TEXT,
                    data TEXT
                );

                CREATE TABLE IF NOT EXISTS daily_summaries (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date TEXT UNIQUE NOT NULL,
                    starting_capital REAL,
                    ending_capital REAL,
                    total_pnl REAL DEFAULT 0,
                    trade_count INTEGER DEFAULT 0,
                    win_count INTEGER DEFAULT 0,
                    loss_count INTEGER DEFAULT 0,
                    best_trade REAL DEFAULT 0,
                    worst_trade REAL DEFAULT 0,
                    total_fees REAL DEFAULT 0,
                    strategies_used TEXT,
                    regimes_seen TEXT
                );

                CREATE INDEX IF NOT EXISTS idx_trades_symbol ON trades(symbol);
                CREATE INDEX IF NOT EXISTS idx_trades_strategy ON trades(strategy_name);
                CREATE INDEX IF NOT EXISTS idx_trades_entry_time ON trades(entry_time);
                CREATE INDEX IF NOT EXISTS idx_trades_status ON trades(status);
                CREATE INDEX IF NOT EXISTS idx_equity_timestamp ON equity_snapshots(timestamp);
                CREATE INDEX IF NOT EXISTS idx_events_type ON events(event_type);
                CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
            """)
        _log.info("Trade database initialized at %s", self.db_path)

    # --- Trade Operations ---

    def record_trade_open(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        strategy: str,
        regime: str,
        confidence: float,
        sl: float,
        tp: float,
        trailing: float,
    ) -> int:
        """Record a new trade opening and return the database row ID.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Trade direction — ``"buy"`` or ``"sell"``.
            entry_price: Execution price at entry.
            quantity: Asset quantity purchased or sold.
            strategy: Name of the strategy that generated the signal.
            regime: Market regime label at entry time.
            confidence: Strategy signal confidence in the range [0, 1].
            sl: Stop-loss price level.
            tp: Take-profit price level.
            trailing: Initial trailing-stop price level.

        Returns:
            The auto-incremented primary key (``id``) of the new row.
        """
        with self._conn() as conn:
            cursor = conn.execute(
                """
                INSERT INTO trades (symbol, side, entry_price, quantity,
                    entry_time, strategy_name, regime, confidence,
                    stop_loss, take_profit, trailing_stop, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
            """,
                (
                    symbol,
                    side,
                    entry_price,
                    quantity,
                    datetime.now().isoformat(),
                    strategy,
                    regime,
                    confidence,
                    sl,
                    tp,
                    trailing,
                ),
            )
            return cursor.lastrowid

    def record_trade_close(
        self,
        trade_id: int | None = None,
        symbol: str | None = None,
        side: str | None = None,
        exit_price: float = 0,
        pnl_gross: float = 0,
        pnl_net: float = 0,
        fees: float = 0,
        slippage: float = 0,
        reason: str = "",
        hold_bars: int = 0,
    ) -> None:
        """Record a trade closing by trade ID or by symbol/side lookup.

        Prefer passing *trade_id* for an exact match.  When only *symbol*
        and *side* are supplied, the most recently opened matching trade is
        closed (fallback path).

        Args:
            trade_id: Primary key of the trade to close.  Takes precedence
                over symbol/side when provided.
            symbol: Trading pair symbol used for the fallback lookup.
            side: Trade direction used for the fallback lookup.
            exit_price: Execution price at exit.
            pnl_gross: Gross profit/loss before fees and slippage.
            pnl_net: Net profit/loss after fees and slippage.
            fees: Total fees paid on the round trip.
            slippage: Total slippage cost on the round trip.
            reason: Human-readable exit reason (e.g. ``"stop_loss"``).
            hold_bars: Number of OHLCV bars the position was held.
        """
        with self._conn() as conn:
            if trade_id:
                conn.execute(
                    """
                    UPDATE trades SET exit_price=?, pnl_gross=?, pnl_net=?,
                        fees_paid=?, slippage_cost=?, exit_time=?,
                        exit_reason=?, hold_bars=?, status='closed'
                    WHERE id=?
                """,
                    (
                        exit_price,
                        pnl_gross,
                        pnl_net,
                        fees,
                        slippage,
                        datetime.now().isoformat(),
                        reason,
                        hold_bars,
                        trade_id,
                    ),
                )
            elif symbol and side:
                # Close by symbol/side (fallback) — uses subquery because
                # standard SQLite doesn't support ORDER BY in UPDATE
                conn.execute(
                    """
                    UPDATE trades SET exit_price=?, pnl_gross=?, pnl_net=?,
                        fees_paid=?, slippage_cost=?, exit_time=?,
                        exit_reason=?, hold_bars=?, status='closed'
                    WHERE id = (
                        SELECT id FROM trades
                        WHERE symbol=? AND side=? AND status='open'
                        ORDER BY entry_time DESC LIMIT 1
                    )
                """,
                    (
                        exit_price,
                        pnl_gross,
                        pnl_net,
                        fees,
                        slippage,
                        datetime.now().isoformat(),
                        reason,
                        hold_bars,
                        symbol,
                        side,
                    ),
                )

    def get_open_trades(self) -> list[dict[str, Any]]:
        """Return all currently open trades ordered by entry time.

        Returns:
            List of trade row dictionaries with ``status='open'``.
        """
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE status='open' ORDER BY entry_time").fetchall()
            return [dict(r) for r in rows]

    def orphan_trades(self, trade_ids: list[int]) -> None:
        """Mark specific open trades as abandoned due to a restart.

        Sets ``status='abandoned'`` and ``exit_reason='orphaned_on_restart'``
        for each supplied trade ID.  Called by the startup reconciliation
        routine to prevent stale open positions after a crash.

        Args:
            trade_ids: List of trade primary keys to mark as abandoned.
                If empty, this method is a no-op.
        """
        if not trade_ids:
            return
        placeholders = ",".join("?" * len(trade_ids))
        with self._conn() as conn:
            conn.execute(
                f"UPDATE trades SET status='abandoned', exit_reason='orphaned_on_restart'"  # noqa: S608
                f" WHERE id IN ({placeholders})",
                trade_ids,
            )

    def get_trade_history(
        self, limit: int = 100, strategy: str | None = None, since: str | None = None
    ) -> list[dict[str, Any]]:
        """Return closed trade history with optional filters.

        Args:
            limit: Maximum number of rows to return (most recent first).
            strategy: When provided, restricts results to trades using this
                strategy name.
            since: ISO-8601 datetime string; restricts results to trades
                whose ``exit_time`` is on or after this value.

        Returns:
            List of closed trade row dictionaries ordered by ``exit_time``
            descending.  Each dict uses the column names from the
            ``trades`` table (e.g. ``pnl_net``, ``strategy_name``).
        """
        query = "SELECT * FROM trades WHERE status='closed'"
        params = []

        if strategy:
            query += " AND strategy_name=?"
            params.append(strategy)
        if since:
            query += " AND exit_time>=?"
            params.append(since)

        query += " ORDER BY exit_time DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # --- Equity Snapshots ---

    def record_equity(
        self, equity: float, capital: float, unrealized_pnl: float, open_positions: int, cycle: int
    ) -> None:
        """Record a point-in-time equity snapshot.

        Args:
            equity: Total portfolio equity (capital + unrealized PnL).
            capital: Available cash balance.
            unrealized_pnl: Sum of open-position mark-to-market PnL.
            open_positions: Number of currently open positions.
            cycle: Agent cycle counter at the time of the snapshot.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO equity_snapshots
                    (timestamp, equity, capital, unrealized_pnl, open_positions, cycle)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), equity, capital, unrealized_pnl, open_positions, cycle),
            )

    def get_equity_curve(self, since: str | None = None, limit: int = 1000) -> list[dict[str, Any]]:
        """Return equity snapshots ordered chronologically.

        Args:
            since: ISO-8601 datetime string; restricts results to snapshots
                on or after this timestamp.
            limit: Maximum number of rows to return.

        Returns:
            List of equity snapshot row dictionaries ordered by ``timestamp``
            ascending (oldest first).
        """
        query = "SELECT * FROM equity_snapshots"
        params = []
        if since:
            query += " WHERE timestamp>=?"
            params.append(since)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in reversed(rows)]

    # --- Events ---

    def record_event(
        self, event_type: str, description: str, severity: str = "info", data: dict[str, Any] | None = None
    ) -> None:
        """Record a structured system event.

        Args:
            event_type: Short category label (e.g. ``"trade_open"``,
                ``"regime_change"``).
            description: Human-readable description of the event.
            severity: Log severity level — ``"info"``, ``"warning"``, or
                ``"error"``.
            data: Optional dictionary of supplementary key/value data stored
                as a JSON string.
        """
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (timestamp, event_type, severity, description, data)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), event_type, severity, description, json.dumps(data or {})),
            )

    def get_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Return recent system events, optionally filtered by type.

        Args:
            event_type: When provided, restricts results to events with this
                ``event_type`` value.
            limit: Maximum number of rows to return (most recent first).

        Returns:
            List of event row dictionaries ordered by ``timestamp`` descending.
        """
        query = "SELECT * FROM events"
        params = []
        if event_type:
            query += " WHERE event_type=?"
            params.append(event_type)
        query += " ORDER BY timestamp DESC LIMIT ?"
        params.append(limit)

        with self._conn() as conn:
            rows = conn.execute(query, params).fetchall()
            return [dict(r) for r in rows]

    # --- Daily Summaries ---

    def generate_daily_summary(self, date: str | None = None) -> dict[str, Any]:
        """Generate or retrieve the daily trade summary for a given date.

        If a summary row already exists for *date* it is returned directly;
        otherwise one is computed from closed trades and persisted.

        Args:
            date: Date string in ``YYYY-MM-DD`` format.  Defaults to today.

        Returns:
            Dictionary with summary keys: ``date``, ``total_pnl``,
            ``trade_count``, ``win_count``, ``loss_count``, ``best_trade``,
            ``worst_trade``, ``total_fees``, ``strategies_used``, and
            ``regimes_seen``.  When a cached row already exists in the
            database, ``starting_capital`` and ``ending_capital`` keys are
            also included (as stored at insert time).  Returns
            ``{"date": date, "trade_count": 0}`` when no trades were closed
            on that date.
        """
        date = date or datetime.now().strftime("%Y-%m-%d")

        with self._conn() as conn:
            # Check if already exists
            existing = conn.execute("SELECT * FROM daily_summaries WHERE date=?", (date,)).fetchone()
            if existing:
                return dict(existing)

            # Generate from trades
            trades = conn.execute(
                """
                SELECT * FROM trades
                WHERE status='closed' AND date(exit_time)=?
            """,
                (date,),
            ).fetchall()

            if not trades:
                return {"date": date, "trade_count": 0}

            trades = [dict(t) for t in trades]
            pnls = [t["pnl_net"] for t in trades]
            wins = [p for p in pnls if p > 0]
            losses = [p for p in pnls if p <= 0]
            strategies = list(set(t["strategy_name"] for t in trades if t["strategy_name"]))
            regimes = list(set(t["regime"] for t in trades if t["regime"]))

            summary = {
                "date": date,
                "total_pnl": round(sum(pnls), 2),
                "trade_count": len(trades),
                "win_count": len(wins),
                "loss_count": len(losses),
                "best_trade": round(max(pnls), 2) if pnls else 0,
                "worst_trade": round(min(pnls), 2) if pnls else 0,
                "total_fees": round(sum(t["fees_paid"] for t in trades), 2),
                "strategies_used": json.dumps(strategies),
                "regimes_seen": json.dumps(regimes),
            }

            # Store
            conn.execute(
                """
                INSERT OR REPLACE INTO daily_summaries
                    (date, total_pnl, trade_count, win_count, loss_count,
                     best_trade, worst_trade, total_fees,
                     strategies_used, regimes_seen)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    date,
                    summary["total_pnl"],
                    summary["trade_count"],
                    summary["win_count"],
                    summary["loss_count"],
                    summary["best_trade"],
                    summary["worst_trade"],
                    summary["total_fees"],
                    summary["strategies_used"],
                    summary["regimes_seen"],
                ),
            )

            return summary

    # --- Analytics ---

    def get_strategy_performance(self) -> dict[str, dict[str, Any]]:
        """Return closed-trade performance metrics grouped by strategy.

        Returns:
            Dictionary keyed by ``strategy_name`` where each value contains
            aggregated stats: ``total_trades``, ``wins``, ``total_pnl``,
            ``avg_pnl``, ``best_trade``, ``worst_trade``, ``avg_hold``, and
            ``total_fees``.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT strategy_name,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_net) as total_pnl,
                    AVG(pnl_net) as avg_pnl,
                    MAX(pnl_net) as best_trade,
                    MIN(pnl_net) as worst_trade,
                    AVG(hold_bars) as avg_hold,
                    SUM(fees_paid) as total_fees
                FROM trades
                WHERE status='closed'
                GROUP BY strategy_name
                ORDER BY total_pnl DESC
            """).fetchall()
            return {r["strategy_name"]: dict(r) for r in rows}

    def get_regime_performance(self) -> dict[str, dict[str, Any]]:
        """Return closed-trade performance metrics grouped by market regime.

        Returns:
            Dictionary keyed by ``regime`` where each value contains
            aggregated stats: ``total_trades``, ``wins``, ``total_pnl``,
            and ``avg_pnl``.
        """
        with self._conn() as conn:
            rows = conn.execute("""
                SELECT regime,
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_net) as total_pnl,
                    AVG(pnl_net) as avg_pnl
                FROM trades
                WHERE status='closed'
                GROUP BY regime
                ORDER BY total_pnl DESC
            """).fetchall()
            return {r["regime"]: dict(r) for r in rows}

    def get_total_stats(self) -> dict[str, Any]:
        """Return aggregate statistics across all closed trades.

        Returns:
            Dictionary with keys: ``total_trades``, ``wins``,
            ``total_pnl``, ``avg_pnl``, ``best_trade``, ``worst_trade``,
            ``total_fees``, ``total_slippage``, ``avg_hold_bars``, and
            ``win_rate`` (as a percentage, e.g. ``62.5`` for 62.5%).
            Returns an empty dict when there are no closed trades.

        Note:
            ``win_rate`` is a percentage value (0–100), not a fraction.
            Divide by 100 before comparing to the 0–1 fraction used in
            the dashboard and ``/status`` snapshot.
        """
        with self._conn() as conn:
            row = conn.execute("""
                SELECT
                    COUNT(*) as total_trades,
                    SUM(CASE WHEN pnl_net > 0 THEN 1 ELSE 0 END) as wins,
                    SUM(pnl_net) as total_pnl,
                    AVG(pnl_net) as avg_pnl,
                    MAX(pnl_net) as best_trade,
                    MIN(pnl_net) as worst_trade,
                    SUM(fees_paid) as total_fees,
                    SUM(slippage_cost) as total_slippage,
                    AVG(hold_bars) as avg_hold_bars
                FROM trades
                WHERE status='closed'
            """).fetchone()
            if row:
                d = dict(row)
                d["win_rate"] = round(d["wins"] / d["total_trades"] * 100, 1) if d["total_trades"] > 0 else 0
                return d
            return {}
