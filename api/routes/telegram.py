"""
Telegram Webhook Route
=======================
Receives incoming updates from Telegram (commands, callback queries)
and forwards them to the TelegramBot handler.

Security: validates the X-Telegram-Bot-Api-Secret-Token header
against the configured TELEGRAM_WEBHOOK_SECRET.
"""

import json
import secrets
from typing import Any

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import Config


def create_router(telegram_bot: Any) -> APIRouter:
    """Create Telegram webhook and status routes."""
    router = APIRouter(prefix="/telegram", tags=["telegram"])

    @router.post("/webhook")
    async def telegram_webhook(request: Request) -> dict[str, Any]:
        """Receive and process incoming Telegram updates."""
        # Verify secret token from Telegram (constant-time comparison prevents timing attacks)
        if Config.TELEGRAM_WEBHOOK_SECRET:
            header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if not secrets.compare_digest(header_secret, Config.TELEGRAM_WEBHOOK_SECRET):
                return JSONResponse(status_code=403, content={"detail": "Invalid secret"})

        # Guard against oversized payloads (Telegram updates are never > a few KB)
        raw = await request.body()
        if len(raw) > 1_048_576:  # 1 MB
            return JSONResponse(status_code=413, content={"detail": "Payload too large"})
        body = json.loads(raw)
        # Handle in a non-blocking way — Telegram expects 200 quickly
        telegram_bot.handle_update(body)
        return {"ok": True}

    @router.get("/status")
    def telegram_status() -> dict[str, Any]:
        """Return Telegram bot configuration and status."""
        return {
            "enabled": telegram_bot.enabled,
            "webhook_url": Config.TELEGRAM_WEBHOOK_URL or None,
            "trade_confirmation": Config.TELEGRAM_TRADE_CONFIRMATION,
            "confirmation_timeout": Config.TELEGRAM_CONFIRMATION_TIMEOUT,
        }

    return router
