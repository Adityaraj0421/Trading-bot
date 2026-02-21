"""
WebSocket Route — Real-time dashboard updates
===============================================
Accepts WebSocket connections and streams live agent data.

Auth: after connecting, client sends {"type":"auth","api_key":"xxx"}
as the first message.  When API_AUTH_KEY is not set, auth is skipped.
This avoids leaking the API key in URL query parameters.
"""

import asyncio
import json

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websocket_manager import WebSocketManager
from config import Config


def create_router(ws_manager: WebSocketManager) -> APIRouter:
    router = APIRouter(tags=["websocket"])

    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket):
        await ws_manager.connect(ws)
        try:
            # --- Auth phase (message-based, not query param) ---
            if Config.API_AUTH_KEY:
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                    msg = json.loads(raw)
                    if (
                        msg.get("type") != "auth"
                        or msg.get("api_key") != Config.API_AUTH_KEY
                    ):
                        await ws.close(code=4001, reason="Invalid API key")
                        return
                except (asyncio.TimeoutError, json.JSONDecodeError, KeyError):
                    await ws.close(code=4001, reason="Auth timeout or invalid format")
                    return

            # --- Keepalive loop ---
            while True:
                data = await ws.receive_text()
                if data == "ping":
                    await ws.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(ws)

    return router
