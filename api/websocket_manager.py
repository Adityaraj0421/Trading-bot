"""
WebSocket Connection Manager
==============================
Manages connected WebSocket clients and broadcasts real-time updates.
Provides a thread-safe sync wrapper for broadcasting from the agent thread.
"""

import asyncio
import json
import logging
from datetime import datetime

from fastapi import WebSocket

_log = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events to all clients."""

    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the FastAPI event loop for thread-safe broadcasting."""
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        """Accept a new WebSocket connection and register it."""
        await ws.accept()
        self._connections.add(ws)
        _log.info("WebSocket client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        """Unregister a WebSocket connection."""
        self._connections.discard(ws)
        _log.info("WebSocket client disconnected (%d total)", len(self._connections))

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Send an event to all connected WebSocket clients."""
        if not self._connections:
            return

        message = json.dumps(
            {
                "type": event_type,
                "data": data,
                "ts": datetime.now().isoformat(),
            }
        )

        dead = set()
        for ws in self._connections:
            try:
                await ws.send_text(message)
            except (ConnectionError, RuntimeError) as e:
                _log.debug("WebSocket send failed (connection): %s", e)
                dead.add(ws)
            except Exception as e:
                _log.warning("WebSocket send unexpected error: %s", e)
                dead.add(ws)

        # Clean up broken connections
        self._connections -= dead

    def broadcast_sync(self, event_type: str, data: dict) -> None:
        """Thread-safe broadcast from the synchronous agent thread.

        Schedules the async broadcast on the FastAPI event loop.
        Safe to call from any thread.
        """
        if not self._connections or not self._loop:
            return
        try:
            asyncio.run_coroutine_threadsafe(
                self.broadcast(event_type, data),
                self._loop,
            )
        except RuntimeError as e:
            _log.debug("Broadcast scheduling error (event loop closed?): %s", e)
        except Exception as e:
            _log.warning("Broadcast scheduling unexpected error: %s", e)

    @property
    def client_count(self) -> int:
        """Return the number of active WebSocket connections."""
        return len(self._connections)
