"""
Status routes — health check, agent status, configuration.

Endpoints:
  GET /health              — Liveness check with staleness detection (public).
  GET /status              — Full agent snapshot enriched with TradeDB stats.
  GET /config              — Current trading configuration parameters.
  GET /system/modules      — v7.0 module enable/status overview.
  GET /system/rate-limiter — Rate limiter usage statistics.
  GET /notifications       — Recent notification log (last N entries).
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from api.data_store import DataStore
from config import Config


def create_router(store: DataStore) -> APIRouter:
    """Create the status router with health, config, and system-module endpoints.

    Args:
        store: The shared DataStore instance used by the agent and API.

    Returns:
        Configured ``APIRouter`` with all status endpoints registered.
    """
    router = APIRouter(tags=["status"])

    @router.get("/health")
    def health() -> Any:
        """Return health check with agent liveness and staleness detection."""
        snapshot = store.get_snapshot()
        last_update = snapshot.get("updated_at") if snapshot else None
        # Detect stale agent: if no update for 5+ minutes, report degraded
        if last_update:
            try:
                age = (datetime.now() - datetime.fromisoformat(last_update)).total_seconds()
                if age > 300:
                    return JSONResponse(
                        status_code=503,
                        content={"status": "degraded", "agent_running": True, "last_update": last_update,
                                 "detail": f"Agent not updating (last update {age:.0f}s ago)"},
                    )
            except (ValueError, TypeError):
                pass  # Malformed timestamp — treat as healthy
        return {
            "status": "healthy",
            "agent_running": bool(snapshot),
            "last_update": last_update,
        }

    @router.get("/status")
    def status() -> dict[str, Any]:
        """Return current agent snapshot, enriched with persistent TradeDB stats."""
        snapshot = store.get_snapshot()
        if not snapshot:
            return {"status": "waiting"}

        # Merge persistent stats from TradeDB when session data is incomplete
        db = store.get_trade_db()
        if db is not None:
            try:
                db_stats = db.get_total_stats()
                if db_stats and db_stats.get("total_trades", 0) > 0:
                    session_trades = snapshot.get("total_trades", 0)
                    # Use TradeDB values if they exceed session-only counts
                    if db_stats["total_trades"] > session_trades:
                        snapshot["total_pnl"] = round(db_stats.get("total_pnl") or 0, 2)
                        snapshot["total_trades"] = db_stats["total_trades"]
                        # DB returns win_rate as percentage; snapshot uses 0-1 fraction
                        snapshot["win_rate"] = round(
                            (db_stats.get("win_rate") or 0) / 100, 4
                        )
                        snapshot["total_fees"] = round(db_stats.get("total_fees") or 0, 2)
            except Exception:
                pass  # Fallback to session data on any DB error

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
                    "providers_enabled": sum(
                        [
                            # Flagged providers (gated by Config flag in get_signal())
                            Config.ENABLE_ONCHAIN,
                            Config.ENABLE_ORDERBOOK,
                            Config.ENABLE_CORRELATION,
                            Config.ENABLE_WHALE_TRACKING,
                            Config.ENABLE_NEWS_NLP,
                            # Flagged providers that always run (no gate in get_signal)
                            Config.ENABLE_FUNDING_OI,
                            Config.ENABLE_LIQUIDATION,
                            # Always-on providers (no ENABLE_ flag, public APIs or
                            # keyword fallback — run regardless of config)
                            True,  # LLMSentimentProvider
                            True,  # CascadePredictor
                            True,  # FearGreedProvider
                        ]
                    ),
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
