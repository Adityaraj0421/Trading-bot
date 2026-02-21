"""
WebSocket Route — Real-time dashboard updates
===============================================
Accepts WebSocket connections and streams live agent data.
Auth via query parameter ?api_key=xxx (WebSocket can't use HTTP headers in middleware).
"""

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, Query

from api.websocket_manager import WebSocketManager
from config import Config


def create_router(ws_manager: WebSocketManager) -> APIRouter:
    router = APIRouter(tags=["websocket"])

    @router.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket, api_key: str = Query(default="")):
        # Auth check for WebSocket connections
        if Config.API_AUTH_KEY and api_key != Config.API_AUTH_KEY:
            await ws.close(code=4001, reason="Invalid API key")
            return

        await ws_manager.connect(ws)
        try:
            # Keep connection alive — listen for pings or client messages
            while True:
                data = await ws.receive_text()
                # Client can send "ping" to keep alive
                if data == "ping":
                    await ws.send_text('{"type":"pong"}')
        except WebSocketDisconnect:
            pass
        finally:
            await ws_manager.disconnect(ws)

    return router
