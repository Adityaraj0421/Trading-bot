"""
WebSocket Route — Real-time dashboard updates
===============================================
Accepts WebSocket connections and streams live agent data.

Auth: after connecting, client sends {"type":"auth","api_key":"xxx"}
as the first message.  When API_AUTH_KEY is not set, auth is skipped.
This avoids leaking the API key in URL query parameters.

The API key comparison uses ``secrets.compare_digest()`` to prevent
timing side-channel attacks.

Endpoints:
  WS /ws — Persistent WebSocket connection for real-time agent events.
"""

from __future__ import annotations

import asyncio
import json
import logging
import secrets

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.websocket_manager import WebSocketManager
from config import Config


def create_router(ws_manager: WebSocketManager) -> APIRouter:
    """Create the WebSocket router for real-time dashboard streaming.

    Args:
        ws_manager: The ``WebSocketManager`` that tracks connections and
            broadcasts events from the agent thread.

    Returns:
        Configured ``APIRouter`` with the ``/ws`` WebSocket endpoint.
    """
    router = APIRouter(tags=["websocket"])

    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """Handle WebSocket connections with message-based auth and keepalive."""
        await ws_manager.connect(ws)
        try:
            # --- Auth phase (message-based, not query param) ---
            if Config.API_AUTH_KEY:
                _ws_log = logging.getLogger(__name__)
                try:
                    raw = await asyncio.wait_for(ws.receive_text(), timeout=5.0)
                    msg = json.loads(raw)
                    if msg.get("type") != "auth" or not secrets.compare_digest(
                        msg.get("api_key", ""), Config.API_AUTH_KEY
                    ):
                        await ws.close(code=4001, reason="Invalid API key")
                        return
                except TimeoutError:
                    _ws_log.debug("WebSocket auth timed out (5s)")
                    await ws.close(code=4001, reason="Auth timeout")
                    return
                except json.JSONDecodeError:
                    _ws_log.debug("WebSocket auth message was not valid JSON")
                    await ws.close(code=4001, reason="Invalid auth format")
                    return
                except KeyError:
                    _ws_log.debug("WebSocket auth message missing required fields")
                    await ws.close(code=4001, reason="Invalid auth format")
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
