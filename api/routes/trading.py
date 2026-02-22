"""
Trading routes — trades, positions, equity history.

Trade and equity data are served from the persistent TradeDB (SQLite) when
available, with an automatic fallback to the in-memory DataStore for the
current session. This ensures trade history and equity curves survive server
restarts.
"""

from typing import Any

from fastapi import APIRouter, Query

from api.data_store import DataStore


def create_router(store: DataStore) -> APIRouter:
    """Create trading routes (trades, positions, equity, PnL)."""
    router = APIRouter(tags=["trading"])

    @router.get("/trades")
    def get_trades(
        limit: int = Query(default=100, ge=0),
        status: str = Query(default="all", pattern="^(all|open|closed)$"),
    ) -> dict[str, Any]:
        """Return trade history, preferring the persistent DB over the in-memory log.

        Args:
            limit: Maximum number of trades to return (0 = unlimited).
            status: Filter by trade status — ``all``, ``open``, or ``closed``.
        """
        db = store.get_trade_db()
        if db is not None:
            fetch_limit = limit if limit > 0 else 10000
            if status == "open":
                trades = db.get_open_trades()
            elif status == "closed":
                trades = db.get_trade_history(limit=fetch_limit)
            else:
                # Merge and sort by entry_time DESC (most recently opened first)
                closed = db.get_trade_history(limit=fetch_limit)
                open_trades = db.get_open_trades()
                trades = sorted(
                    open_trades + closed,
                    key=lambda t: t.get("entry_time", ""),
                    reverse=True,
                )
            total = len(trades)
            if limit > 0:
                trades = trades[:limit]
        else:
            # Fallback: in-memory log (current session only)
            trades = store.get_trade_log()
            total = len(trades)
            if limit > 0:
                trades = trades[-limit:]

        return {"trades": trades, "total": total}

    @router.get("/positions")
    def get_positions() -> dict[str, Any]:
        """Return currently open positions."""
        snapshot = store.get_snapshot()
        positions = snapshot.get("positions", [])
        return {"positions": positions, "count": len(positions)}

    @router.get("/equity")
    def get_equity(limit: int = Query(default=0, ge=0)) -> dict[str, Any]:
        """Return equity curve data, preferring the persistent DB over the in-memory log.

        Args:
            limit: Maximum number of data points (0 = unlimited).
        """
        db = store.get_trade_db()
        if db is not None:
            fetch_limit = limit if limit > 0 else 10000
            rows = db.get_equity_curve(limit=fetch_limit)
            total_points = len(rows)
            # Normalise to the same shape the dashboard expects
            equity = [
                {"equity": row["equity"], "timestamp": row["timestamp"]}
                for row in rows
            ]
        else:
            equity = store.get_equity_history()
            total_points = len(equity)
            if limit > 0:
                equity = equity[-limit:]

        return {"equity": equity, "total_points": total_points}

    @router.get("/pnl-summary")
    def get_pnl_summary() -> dict[str, Any]:
        """PnL breakdown by pair and by strategy."""
        db = store.get_trade_db()
        if db is not None:
            closed = db.get_trade_history(limit=10000)
        else:
            all_trades = store.get_trade_log()
            closed = [t for t in all_trades if t.get("exit_price")]

        # By pair
        by_pair: dict = {}
        for t in closed:
            pair = t.get("symbol", "BTC/USDT")
            if pair not in by_pair:
                by_pair[pair] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
            by_pair[pair]["trades"] += 1
            pnl = t.get("pnl_net", 0)
            by_pair[pair]["pnl"] += pnl
            if pnl >= 0:
                by_pair[pair]["wins"] += 1
            else:
                by_pair[pair]["losses"] += 1

        # By strategy — trade_db uses "strategy_name"; in-memory uses "strategy"
        by_strategy: dict = {}
        for t in closed:
            strat = t.get("strategy_name") or t.get("strategy", "Unknown")
            if strat not in by_strategy:
                by_strategy[strat] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
            by_strategy[strat]["trades"] += 1
            pnl = t.get("pnl_net", 0)
            by_strategy[strat]["pnl"] += pnl
            if pnl >= 0:
                by_strategy[strat]["wins"] += 1
            else:
                by_strategy[strat]["losses"] += 1

        # Cumulative PnL curve (chronological order)
        ordered = sorted(closed, key=lambda t: t.get("exit_time") or t.get("entry_time") or "")
        cum_pnl = []
        running = 0.0
        for t in ordered:
            running += t.get("pnl_net", 0)
            cum_pnl.append(
                {
                    "pnl": round(running, 2),
                    "timestamp": t.get("exit_time", t.get("entry_time", "")),
                    "symbol": t.get("symbol", ""),
                }
            )

        return {
            "by_pair": by_pair,
            "by_strategy": by_strategy,
            "cumulative_pnl": cum_pnl,
            "total_closed": len(closed),
        }

    return router
