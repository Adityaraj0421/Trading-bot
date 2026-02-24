"""
Arbitrage monitoring routes.

Endpoints:
  GET /arbitrage/opportunities — Detected cross-exchange arbitrage opportunities.
  GET /arbitrage/fees          — Fee summary across configured exchanges.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter

from api.data_store import DataStore


def create_router(store: DataStore) -> APIRouter:
    """Create the arbitrage router with opportunity and fee endpoints.

    Args:
        store: The shared DataStore instance used by the agent and API.

    Returns:
        Configured ``APIRouter`` with prefix ``/arbitrage``.
    """
    router = APIRouter(prefix="/arbitrage", tags=["arbitrage"])

    @router.get("/opportunities")
    async def get_opportunities() -> dict[str, Any]:
        """Return detected arbitrage opportunities."""
        arb = store.get_arbitrage()
        if not arb:
            return {"status": "not_enabled", "message": "Enable arbitrage in .env"}
        return arb

    @router.get("/fees")
    async def get_fees() -> dict[str, Any]:
        """Return fee summary across exchanges."""
        from arbitrage.fee_calculator import FeeCalculator

        calc = FeeCalculator()
        return calc.get_fee_summary()

    return router
