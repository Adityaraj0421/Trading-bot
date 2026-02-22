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

import json
import logging
import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime
from typing import Any

from config import Config

_log = logging.getLogger(__name__)


class TradeDB:
    """SQLite-backed trade and performance database."""

    def __init__(self, db_path: str | None = None) -> None:
        self.db_path = db_path or os.path.join(getattr(Config, "DATA_DIR", "."), "trades.db")
        self._init_db()

    @contextmanager
    def _conn(self):
        """Context manager for database connections."""
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
        """Close the database (no-op since we use per-query connections)."""
        _log.info("TradeDB closed: %s", self.db_path)

    def _init_db(self):
        """Create tables if they don't exist."""
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
        """Record a new trade opening. Returns trade ID."""
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
        """Record a trade closing."""
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
        """Get all currently open trades."""
        with self._conn() as conn:
            rows = conn.execute("SELECT * FROM trades WHERE status='open' ORDER BY entry_time").fetchall()
            return [dict(r) for r in rows]

    def get_trade_history(
        self, limit: int = 100, strategy: str | None = None, since: str | None = None
    ) -> list[dict[str, Any]]:
        """Get trade history with optional filters."""
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
        """Record an equity snapshot."""
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
        """Get equity curve data."""
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
        """Record a system event."""
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO events (timestamp, event_type, severity, description, data)
                VALUES (?, ?, ?, ?, ?)
            """,
                (datetime.now().isoformat(), event_type, severity, description, json.dumps(data or {})),
            )

    def get_events(self, event_type: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """Get system events."""
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
        """Generate or retrieve a daily summary."""
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
        """Get performance breakdown by strategy."""
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
        """Get performance breakdown by market regime."""
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
        """Get overall trading statistics."""
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
