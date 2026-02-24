"""
WebSocket Connection Manager
==============================
Manages connected WebSocket clients and broadcasts real-time updates.
Provides a thread-safe sync wrapper for broadcasting from the agent thread.
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime

from fastapi import WebSocket

_log = logging.getLogger(__name__)


class WebSocketManager:
    """Manages WebSocket connections and broadcasts events to all clients."""

    def __init__(self) -> None:
        """Initialize the manager with an empty connection set.

        The event loop reference (``_loop``) must be set via
        ``set_event_loop()`` before ``broadcast_sync()`` can be called from
        the agent thread.
        """
        self._connections: set[WebSocket] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def set_event_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Capture the FastAPI event loop for thread-safe broadcasting.

        Must be called once during FastAPI lifespan startup so that
        ``broadcast_sync()`` can schedule coroutines on the correct loop.

        Args:
            loop: The running asyncio event loop from
                ``asyncio.get_running_loop()``.
        """
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        """Accept a new WebSocket connection and register it for broadcasting.

        Args:
            ws: The incoming ``WebSocket`` instance from the FastAPI route.
        """
        await ws.accept()
        self._connections.add(ws)
        _log.info("WebSocket client connected (%d total)", len(self._connections))

    async def disconnect(self, ws: WebSocket) -> None:
        """Unregister a WebSocket connection after disconnect or error.

        Args:
            ws: The ``WebSocket`` instance to remove from the active set.
        """
        self._connections.discard(ws)
        _log.info("WebSocket client disconnected (%d total)", len(self._connections))

    async def broadcast(self, event_type: str, data: dict) -> None:
        """Send a typed event to all connected WebSocket clients.

        Dead connections are detected and removed automatically. The message
        is serialized as JSON with ``type``, ``data``, and ``ts`` fields.

        Args:
            event_type: Event category string (e.g. ``"snapshot"``,
                ``"trade"``, ``"equity"``, ``"event"``).
            data: Payload dict to include in the message.
        """
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
        """Thread-safe broadcast callable for use from the synchronous agent thread.

        Schedules ``broadcast()`` as a coroutine on the FastAPI event loop via
        ``asyncio.run_coroutine_threadsafe``. No-ops if no clients are
        connected or if the event loop has not been set.

        Args:
            event_type: Event category string (e.g. ``"snapshot"``,
                ``"trade"``, ``"equity"``, ``"event"``).
            data: Payload dict to broadcast to all connected clients.
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
