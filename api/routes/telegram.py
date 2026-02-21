"""
Telegram Webhook Route
=======================
Receives incoming updates from Telegram (commands, callback queries)
and forwards them to the TelegramBot handler.

Security: validates the X-Telegram-Bot-Api-Secret-Token header
against the configured TELEGRAM_WEBHOOK_SECRET.
"""

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from config import Config


def create_router(telegram_bot) -> APIRouter:
    router = APIRouter(prefix="/telegram", tags=["telegram"])

    @router.post("/webhook")
    async def telegram_webhook(request: Request):
        # Verify secret token from Telegram
        if Config.TELEGRAM_WEBHOOK_SECRET:
            header_secret = request.headers.get("X-Telegram-Bot-Api-Secret-Token", "")
            if header_secret != Config.TELEGRAM_WEBHOOK_SECRET:
                return JSONResponse(status_code=403, content={"detail": "Invalid secret"})

        body = await request.json()
        # Handle in a non-blocking way — Telegram expects 200 quickly
        telegram_bot.handle_update(body)
        return {"ok": True}

    @router.get("/status")
    def telegram_status():
        return {
            "enabled": telegram_bot.enabled,
            "webhook_url": Config.TELEGRAM_WEBHOOK_URL or None,
            "trade_confirmation": Config.TELEGRAM_TRADE_CONFIRMATION,
            "confirmation_timeout": Config.TELEGRAM_CONFIRMATION_TIMEOUT,
        }

    return router
