"""Backtest routes — run and view backtest results (v2.3)."""

import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.data_store import DataStore

# Track active backtest thread to prevent resource exhaustion (mutable container for closure)
_state = {"active_thread": None}

# Valid values for constrained fields
_VALID_SCENARIOS = {"bull_run", "bear_market", "sideways_chop", "flash_crash", "black_swan", "accumulation"}
_VALID_TIMEFRAMES = {"1m", "3m", "5m", "15m", "30m", "1h", "2h", "4h", "6h", "8h", "12h", "1d", "3d", "1w", "1M"}


class BacktestRequest(BaseModel):
    pair: str | None = Field(
        default=None,
        pattern=r"^[A-Z0-9]{2,10}/[A-Z0-9]{2,10}$",
        description="Trading pair (e.g., BTC/USDT)",
    )
    scenario: str | None = Field(
        default=None,
        max_length=50,
        description="Named scenario (use /backtest/scenarios for valid list)",
    )
    timeframe: str | None = Field(
        default=None,
        pattern=r"^[0-9]{1,2}[mhdwM]$",
        description="Candle timeframe (e.g., 1h, 4h, 1d)",
    )
    periods: int = Field(default=500, gt=0, le=10000, description="Number of bars to backtest")
    mode: str | None = Field(default=None, pattern="^(all_pairs|all_scenarios|all_timeframes)?$")


def create_router(store: DataStore) -> APIRouter:
    """Create backtest routes (run, results, scenarios)."""
    router = APIRouter(prefix="/backtest", tags=["backtest"])

    @router.post("/run")
    async def run_backtest(req: BacktestRequest) -> dict[str, Any]:
        """Run a backtest in a background thread (max 1 concurrent)."""
        # Rate limit: reject if a backtest is already running
        active = _state["active_thread"]
        if active is not None and active.is_alive():
            return {"status": "rejected", "message": "A backtest is already running. Wait for it to finish."}

        def _run():
            import logging

            log = logging.getLogger(__name__)
            from backtest_runner import BacktestRunner

            runner = BacktestRunner()
            try:
                if req.mode == "all_pairs":
                    results = runner.run_multi_pair()
                    existing = store.get_backtest_results()
                    existing.extend(results)
                    store.update_backtest_results(existing)
                elif req.mode == "all_timeframes":
                    results = runner.run_multi_timeframe()
                    existing = store.get_backtest_results()
                    existing.extend(results)
                    store.update_backtest_results(existing)
                elif req.scenario:
                    result = runner.run_scenario(req.scenario, periods=req.periods)
                    existing = store.get_backtest_results()
                    existing.append(result)
                    store.update_backtest_results(existing)
                elif req.pair:
                    results = runner.run_multi_pair([req.pair], limit=req.periods)
                    existing = store.get_backtest_results()
                    existing.extend(results)
                    store.update_backtest_results(existing)
                else:
                    results = runner.run_all_scenarios()
                    existing = store.get_backtest_results()
                    existing.extend(results)
                    store.update_backtest_results(existing)
            except Exception as e:
                log.error("Backtest error: %s", e, exc_info=True)

        _state["active_thread"] = threading.Thread(target=_run, daemon=True)
        _state["active_thread"].start()
        return {"status": "started", "message": "Backtest running in background"}

    @router.post("/clear")
    async def clear_results() -> dict[str, Any]:
        """Clear all backtest results."""
        store.update_backtest_results([])
        return {"status": "cleared"}

    @router.get("/results")
    async def get_results() -> dict[str, Any]:
        """Return stored backtest results."""
        results = store.get_backtest_results()
        return {"results": results, "total": len(results)}

    @router.get("/scenarios")
    async def get_scenarios() -> dict[str, Any]:
        """List available backtest scenarios."""
        from scenarios import list_scenarios as ls

        return {"scenarios": ls()}

    return router
