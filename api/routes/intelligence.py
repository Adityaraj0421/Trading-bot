"""Intelligence signal routes."""

from typing import Any

from fastapi import APIRouter

from api.data_store import DataStore


def create_router(store: DataStore) -> APIRouter:
    """Create intelligence signal routes."""
    router = APIRouter(prefix="/intelligence", tags=["intelligence"])

    @router.get("/signals")
    async def get_signals() -> dict[str, Any]:
        """Return aggregated intelligence signals from all providers."""
        signals = store.get_intelligence()
        if not signals:
            from config import Config

            if Config.any_intelligence_enabled():
                return {"status": "awaiting_first_cycle", "message": "Waiting for agent to complete first cycle"}
            return {"status": "not_enabled", "message": "Enable intelligence toggles in .env"}
        return signals

    @router.get("/providers")
    async def list_providers() -> dict[str, Any]:
        """List intelligence providers and their enabled status."""
        from config import Config

        return {
            "providers": [
                {"name": "onchain", "enabled": Config.ENABLE_ONCHAIN},
                {"name": "orderbook", "enabled": Config.ENABLE_ORDERBOOK},
                {"name": "correlation", "enabled": Config.ENABLE_CORRELATION},
                {"name": "whale_tracker", "enabled": Config.ENABLE_WHALE_TRACKING},
                {"name": "news_sentiment", "enabled": Config.ENABLE_NEWS_NLP},
            ]
        }

    return router
