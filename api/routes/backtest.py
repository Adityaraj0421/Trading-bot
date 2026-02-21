"""Backtest routes — run and view backtest results (v2.2)."""
import threading
from fastapi import APIRouter
from pydantic import BaseModel, Field
from typing import Optional
from api.data_store import DataStore

# Track active backtest thread to prevent resource exhaustion (mutable container for closure)
_state = {"active_thread": None}


class BacktestRequest(BaseModel):
    pair: Optional[str] = None
    scenario: Optional[str] = None
    timeframe: Optional[str] = None
    periods: int = Field(default=500, gt=0, le=10000, description="Number of bars to backtest")
    mode: Optional[str] = Field(default=None, pattern="^(all_pairs|all_scenarios|all_timeframes)?$")


def create_router(store: DataStore) -> APIRouter:
    router = APIRouter(prefix="/backtest", tags=["backtest"])

    @router.post("/run")
    async def run_backtest(req: BacktestRequest):
        """Run a backtest in a background thread (max 1 concurrent)."""
        # Rate limit: reject if a backtest is already running
        active = _state["active_thread"]
        if active is not None and active.is_alive():
            return {"status": "rejected", "message": "A backtest is already running. Wait for it to finish."}

        def _run():
            import logging
            log = logging.getLogger("backtest.api")
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
    async def clear_results():
        """Clear all backtest results."""
        store.update_backtest_results([])
        return {"status": "cleared"}

    @router.get("/results")
    async def get_results():
        results = store.get_backtest_results()
        return {"results": results, "total": len(results)}

    @router.get("/scenarios")
    async def get_scenarios():
        from scenarios import list_scenarios as ls
        return {"scenarios": ls()}

    return router
