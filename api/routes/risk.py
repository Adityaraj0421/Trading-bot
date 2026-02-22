"""Risk simulation routes — Monte Carlo, VaR, stress testing (v2.0)."""
import threading
from typing import Any

from fastapi import APIRouter
from pydantic import BaseModel, Field

from api.data_store import DataStore

# Track active simulation thread (mutable container for closure — avoids nonlocal on module var)
_state = {"active_thread": None}


class MonteCarloRequest(BaseModel):
    n_simulations: int | None = Field(default=None, gt=0, le=100000, description="Max 100K simulations")
    n_days: int | None = Field(default=None, gt=0, le=1000, description="Max 1000 days horizon")
    initial_equity: float | None = Field(default=None, gt=0, le=1e9, description="Starting equity")


def create_router(store: DataStore) -> APIRouter:
    router = APIRouter(prefix="/risk", tags=["risk"])

    @router.get("/simulation")
    async def get_simulation() -> dict[str, Any]:
        mc = store.get_monte_carlo()
        if not mc:
            return {"status": "not_run", "message": "Run Monte Carlo simulation first"}
        return mc

    @router.post("/monte-carlo")
    async def run_monte_carlo(req: MonteCarloRequest) -> dict[str, Any]:
        """Run Monte Carlo simulation in background thread (max 1 concurrent)."""
        active = _state["active_thread"]
        if active is not None and active.is_alive():
            return {"status": "rejected", "message": "A simulation is already running."}

        def _run():
            from risk_simulation.monte_carlo import MonteCarloSimulator
            from risk_simulation.scenarios import StressTestRunner
            from config import Config

            # Get trade returns from backtest results or trade log
            trade_log = store.get_trade_log()
            if trade_log:
                returns = []
                for t in trade_log:
                    entry = t.get("entry_price", 0)
                    if entry > 0:
                        pnl_pct = t.get("pnl_net", 0) / (entry * t.get("quantity", 1))
                        returns.append(pnl_pct)
            else:
                # Use synthetic returns for demo
                import numpy as np
                returns = list(np.random.normal(0.001, 0.02, 100))

            sim = MonteCarloSimulator(
                n_simulations=req.n_simulations,
                n_days=req.n_days,
            )
            equity = req.initial_equity or Config.INITIAL_CAPITAL
            result = sim.run(returns, initial_equity=equity)

            # Run stress tests too
            stress = StressTestRunner()
            stress_results = stress.run_stress_test(equity)

            store.update_monte_carlo({
                "monte_carlo": result.to_dict(),
                "stress_tests": stress_results,
                "status": "completed",
            })

        _state["active_thread"] = threading.Thread(target=_run, daemon=True)
        _state["active_thread"].start()
        return {"status": "started", "message": "Monte Carlo simulation running in background"}

    @router.get("/stress-tests")
    async def get_stress_scenarios() -> dict[str, Any]:
        from risk_simulation.scenarios import StressTestRunner
        runner = StressTestRunner()
        return {"scenarios": runner.list_scenarios()}

    return router
