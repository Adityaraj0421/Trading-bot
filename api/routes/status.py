"""
Status routes — health check, agent status, configuration.
"""

from typing import Any

from fastapi import APIRouter

from api.data_store import DataStore
from config import Config


def create_router(store: DataStore) -> APIRouter:
    """Create status routes (health, config, system modules)."""
    router = APIRouter(tags=["status"])

    @router.get("/health")
    def health() -> dict[str, Any]:
        """Return health check with agent liveness."""
        snapshot = store.get_snapshot()
        return {
            "status": "healthy",
            "agent_running": bool(snapshot),
            "last_update": snapshot.get("updated_at"),
        }

    @router.get("/status")
    def status() -> dict[str, Any]:
        """Return current agent snapshot (cycle data, positions, PnL)."""
        snapshot = store.get_snapshot()
        if not snapshot:
            return {"status": "waiting"}
        return snapshot

    @router.get("/config")
    def config() -> dict[str, Any]:
        """Return current agent configuration (pairs, timeframe, risk params)."""
        return {
            "exchange": Config.EXCHANGE_ID,
            "pair": Config.TRADING_PAIR,
            "pairs": Config.TRADING_PAIRS,
            "timeframe": Config.TIMEFRAME,
            "confirmation_timeframe": Config.CONFIRMATION_TIMEFRAME,
            "mode": Config.TRADING_MODE,
            "initial_capital": Config.INITIAL_CAPITAL,
            "max_position_pct": Config.MAX_POSITION_PCT,
            "stop_loss_pct": Config.STOP_LOSS_PCT,
            "take_profit_pct": Config.TAKE_PROFIT_PCT,
            "trailing_stop_pct": Config.TRAILING_STOP_PCT,
            "fee_pct": Config.FEE_PCT,
            "slippage_pct": Config.SLIPPAGE_PCT,
            "agent_interval_seconds": Config.AGENT_INTERVAL_SECONDS,
            # v7.0
            "enable_websocket": Config.ENABLE_WEBSOCKET,
            "enable_trade_db": Config.ENABLE_TRADE_DB,
            "notifications_enabled": Config.any_notifications_enabled(),
            "intelligence_enabled": Config.any_intelligence_enabled(),
            "max_requests_per_minute": Config.MAX_REQUESTS_PER_MINUTE,
            "max_orders_per_minute": Config.MAX_ORDERS_PER_MINUTE,
        }

    @router.get("/system/modules")
    def system_modules():
        """Return status of all v7.0 system modules."""
        modules = store.get_system_modules()
        if not modules:
            return {
                "websocket": {"enabled": Config.ENABLE_WEBSOCKET, "status": "unknown"},
                "notifications": {
                    "enabled": Config.any_notifications_enabled(),
                    "channels": {
                        "telegram": bool(Config.TELEGRAM_BOT_TOKEN and Config.TELEGRAM_CHAT_ID),
                        "discord": bool(Config.DISCORD_WEBHOOK_URL),
                        "email": Config.EMAIL_ALERTS_ENABLED,
                    },
                },
                "trade_db": {"enabled": Config.ENABLE_TRADE_DB, "status": "unknown"},
                "rate_limiter": {
                    "max_requests_per_minute": Config.MAX_REQUESTS_PER_MINUTE,
                    "max_orders_per_minute": Config.MAX_ORDERS_PER_MINUTE,
                },
                "intelligence": {
                    "enabled": Config.any_intelligence_enabled(),
                    "providers_enabled": sum([
                        Config.ENABLE_ONCHAIN, Config.ENABLE_ORDERBOOK,
                        Config.ENABLE_CORRELATION, Config.ENABLE_WHALE_TRACKING,
                        Config.ENABLE_NEWS_NLP,
                    ]),
                },
            }
        return modules

    @router.get("/system/rate-limiter")
    def rate_limiter_stats():
        """Return rate limiter usage statistics."""
        stats = store.get_rate_limiter_stats()
        if not stats:
            return {
                "max_requests_per_minute": Config.MAX_REQUESTS_PER_MINUTE,
                "max_orders_per_minute": Config.MAX_ORDERS_PER_MINUTE,
                "requests_used": 0,
                "orders_used": 0,
            }
        return stats

    @router.get("/notifications")
    def notifications(limit: int = 100):
        """Return recent notification log entries."""
        notifs = store.get_notifications(limit=limit)
        return {"notifications": notifs, "count": len(notifs)}

    return router
