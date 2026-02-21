"""
Trading routes — trades, positions, equity history.
"""

from fastapi import APIRouter, Query

from api.data_store import DataStore


def create_router(store: DataStore) -> APIRouter:
    router = APIRouter(tags=["trading"])

    @router.get("/trades")
    def get_trades(limit: int = Query(default=100, ge=0)):
        trades = store.get_trade_log()
        total = len(trades)
        if limit > 0:
            trades = trades[-limit:]
        return {"trades": trades, "total": total}

    @router.get("/positions")
    def get_positions():
        snapshot = store.get_snapshot()
        positions = snapshot.get("positions", [])
        return {"positions": positions, "count": len(positions)}

    @router.get("/equity")
    def get_equity(limit: int = Query(default=0, ge=0)):
        equity = store.get_equity_history()
        total_points = len(equity)
        if limit > 0:
            equity = equity[-limit:]
        return {"equity": equity, "total_points": total_points}

    @router.get("/pnl-summary")
    def get_pnl_summary():
        """v7.0: PnL breakdown by pair and by strategy."""
        trades = store.get_trade_log()
        closed = [t for t in trades if t.get("exit_price")]

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

        # By strategy
        by_strategy: dict = {}
        for t in closed:
            strat = t.get("strategy", "Unknown")
            if strat not in by_strategy:
                by_strategy[strat] = {"trades": 0, "pnl": 0.0, "wins": 0, "losses": 0}
            by_strategy[strat]["trades"] += 1
            pnl = t.get("pnl_net", 0)
            by_strategy[strat]["pnl"] += pnl
            if pnl >= 0:
                by_strategy[strat]["wins"] += 1
            else:
                by_strategy[strat]["losses"] += 1

        # Cumulative PnL curve
        cum_pnl = []
        running = 0.0
        for t in closed:
            running += t.get("pnl_net", 0)
            cum_pnl.append({
                "pnl": round(running, 2),
                "timestamp": t.get("exit_time", t.get("entry_time", "")),
                "symbol": t.get("symbol", ""),
            })

        return {
            "by_pair": by_pair,
            "by_strategy": by_strategy,
            "cumulative_pnl": cum_pnl,
            "total_closed": len(closed),
        }

    return router
