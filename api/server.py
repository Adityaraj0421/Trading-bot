"""
FastAPI Server for the Crypto Trading Agent
=============================================
Exposes the agent's internal state, trades, equity, and configuration
through a REST API.  The agent runs in a background thread and pushes
data into a shared DataStore; the API reads from that store.

Security: API key authentication is enforced when API_AUTH_KEY is set
in the environment. All endpoints except /health require the key.
"""

import asyncio
import logging
import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

_log = logging.getLogger(__name__)

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from api.data_store import DataStore
from api.routes import status, trading, autonomous, backtest, intelligence, arbitrage, risk, telegram
from api.routes import websocket as ws_route
from api.websocket_manager import WebSocketManager
from config import Config
from telegram_bot import TelegramBot


# ---------------------------------------------------------------------------
# API Key Authentication Middleware
# ---------------------------------------------------------------------------
class APIKeyAuthMiddleware(BaseHTTPMiddleware):
    """Validates API key header on all requests except /health and /docs.

    When Config.API_AUTH_KEY is empty, authentication is disabled (dev mode).
    Set API_AUTH_KEY in .env to enable authentication for production.
    """

    # Endpoints that don't require authentication
    PUBLIC_PATHS = {"/health", "/docs", "/openapi.json", "/redoc", "/telegram/webhook"}

    async def dispatch(self, request: Request, call_next: Any) -> Any:
        # Skip auth if no key configured (development mode)
        if not Config.API_AUTH_KEY:
            return await call_next(request)

        # Allow public endpoints without auth
        if request.url.path in self.PUBLIC_PATHS:
            return await call_next(request)

        # Check for API key in header
        api_key = request.headers.get("X-API-Key", "")
        if api_key != Config.API_AUTH_KEY:
            return JSONResponse(
                status_code=401,
                content={"detail": "Invalid or missing API key. Set X-API-Key header."},
            )

        return await call_next(request)

# ---------------------------------------------------------------------------
# Global DataStore shared between the agent thread and API request handlers
# ---------------------------------------------------------------------------
data_store = DataStore()

# ---------------------------------------------------------------------------
# Telegram Bot — interactive command handling + trade confirmations
# ---------------------------------------------------------------------------
telegram_bot = TelegramBot(data_store=data_store)

# ---------------------------------------------------------------------------
# WebSocket Manager — real-time dashboard updates
# ---------------------------------------------------------------------------
ws_manager = WebSocketManager()


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------
def create_app(lifespan=None) -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Crypto Trading Agent API",
        description="Real-time API for the autonomous crypto trading agent",
        version="1.0.0",
        lifespan=lifespan,
    )

    # Authentication middleware (must be added before CORS)
    app.add_middleware(APIKeyAuthMiddleware)

    # CORS — origins from env (CORS_ORIGINS), defaults to localhost:3000
    origins = [o.strip() for o in Config.CORS_ORIGINS.split(",") if o.strip()]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-API-Key"],
    )

    # Register route modules — each gets a reference to the shared data store
    app.include_router(status.create_router(data_store))
    app.include_router(trading.create_router(data_store))
    app.include_router(autonomous.create_router(data_store))
    app.include_router(backtest.create_router(data_store))
    app.include_router(intelligence.create_router(data_store))
    app.include_router(arbitrage.create_router(data_store))
    app.include_router(risk.create_router(data_store))
    app.include_router(telegram.create_router(telegram_bot))
    app.include_router(ws_route.create_router(ws_manager))

    return app


# ---------------------------------------------------------------------------
# Agent thread
# ---------------------------------------------------------------------------
def _run_agent() -> None:
    """Import and start the trading agent in the current thread."""
    try:
        from agent import TradingAgent, set_data_store
        agent = TradingAgent()
        agent.telegram_bot = telegram_bot
        set_data_store(data_store, agent=agent)
        agent.run()
    except Exception as e:
        _log.error("[API] Agent thread error: %s", e, exc_info=True)


# ---------------------------------------------------------------------------
# Lifespan — starts the agent thread on server startup
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Start the trading agent in a daemon thread when the server boots."""
    # Capture the event loop for thread-safe WebSocket broadcasts
    ws_manager.set_event_loop(asyncio.get_running_loop())
    data_store.set_broadcast_callback(ws_manager.broadcast_sync)
    _log.info("[API] WebSocket broadcast wired")

    # Setup Telegram webhook
    if telegram_bot.enabled and Config.TELEGRAM_WEBHOOK_URL:
        ok = telegram_bot.setup_webhook()
        _log.info("[API] Telegram webhook: %s", "registered" if ok else "failed")

    _log.info("[API] Starting trading agent in background thread...")
    agent_thread = threading.Thread(target=_run_agent, daemon=True, name="agent-thread")
    agent_thread.start()
    yield
    # Teardown Telegram webhook
    telegram_bot.teardown_webhook()
    _log.info("[API] Server shutting down.")


# ---------------------------------------------------------------------------
# Application instance used by uvicorn
# ---------------------------------------------------------------------------
app = create_app(lifespan=lifespan)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "api.server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
    )
