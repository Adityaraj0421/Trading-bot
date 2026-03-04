"""WsFeed — Binance WebSocket 1h kline feed.

Runs in a dedicated daemon thread. Subscribes to the Binance combined
stream for all configured symbols and caches the latest closed-candle
close price per symbol.

Usage::

    feed = WsFeed(Config.TRADING_PAIRS)
    feed.start()          # spawns daemon thread
    # in agent cycle:
    price = feed.get_latest_close("BTC/USDT")  # None if no data yet
    feed.stop()
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any

import aiohttp

_log = logging.getLogger(__name__)

_BINANCE_WS_URL = "wss://stream.binance.com:9443/stream"
_MAX_BACKOFF_S = 60.0


def _next_backoff(current: float) -> float:
    """Double ``current`` delay, capped at ``_MAX_BACKOFF_S``."""
    return min(current * 2, _MAX_BACKOFF_S)


class WsFeed:
    """Subscribe to Binance 1h kline WebSocket streams.

    Args:
        symbols: List of trading pairs, e.g. ``["BTC/USDT", "ETH/USDT"]``.
    """

    def __init__(self, symbols: list[str]) -> None:
        self._symbols = symbols
        self._cache: dict[str, float] = {}  # "BTCUSDT" -> latest close
        self._running = False
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------
    # Public API (called from agent thread — all thread-safe)
    # ------------------------------------------------------------------

    def start(self) -> None:
        """Start the WebSocket feed in a background daemon thread."""
        self._running = True
        self._thread = threading.Thread(
            target=self._run_loop, daemon=True, name="ws_feed"
        )
        self._thread.start()
        _log.info("WsFeed started for %s", self._symbols)

    def stop(self) -> None:
        """Signal the feed loop to stop."""
        self._running = False

    def get_latest_close(self, symbol: str) -> float | None:
        """Return the latest closed 1h candle close price for ``symbol``.

        Args:
            symbol: Trading pair in either ``"BTC/USDT"`` or ``"BTCUSDT"`` form.

        Returns:
            Latest close price, or ``None`` if no data has been received yet.
        """
        normalized = symbol.replace("/", "").upper()
        return self._cache.get(normalized)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _stream_names(self) -> list[str]:
        """Return Binance stream names for all configured symbols.

        Returns:
            List of stream name strings like ``["btcusdt@kline_1h", ...]``.
        """
        return [f"{s.replace('/', '').lower()}@kline_1h" for s in self._symbols]

    def _handle(self, data: dict[str, Any]) -> None:
        """Parse a raw WebSocket message and update the cache.

        Args:
            data: Parsed JSON dict from the Binance combined stream.
        """
        stream: str = data.get("stream", "")
        symbol_raw = stream.split("@")[0].upper()  # "btcusdt" -> "BTCUSDT"
        kline: dict = data.get("data", {}).get("k", {})
        if kline.get("x"):  # only closed candles
            self._cache[symbol_raw] = float(kline["c"])
            _log.debug("WsFeed cached %s close=%.2f", symbol_raw, self._cache[symbol_raw])

    def _run_loop(self) -> None:
        """Entry point for the daemon thread — owns its own asyncio loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._connect())
        finally:
            loop.close()

    async def _connect(self) -> None:
        """Connect to Binance WebSocket with exponential-backoff reconnect."""
        streams = "/".join(self._stream_names())
        url = f"{_BINANCE_WS_URL}?streams={streams}"
        retry_delay = 1.0

        while self._running:
            try:
                async with aiohttp.ClientSession() as session, session.ws_connect(url) as ws:
                    _log.info("WsFeed connected to %s", url)
                    retry_delay = 1.0  # reset on successful connect
                    async for msg in ws:
                        if not self._running:
                            break
                        if msg.type == aiohttp.WSMsgType.TEXT:
                            try:
                                self._handle(json.loads(msg.data))
                            except Exception as e:
                                _log.debug("WsFeed parse error: %s", e)
                        elif msg.type in (
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.ERROR,
                        ):
                            _log.warning("WsFeed message type %s — reconnecting", msg.type)
                            break
            except Exception as e:
                _log.warning("WsFeed disconnected: %s — retry in %.0fs", e, retry_delay)

            if self._running:
                await asyncio.sleep(retry_delay)
                retry_delay = _next_backoff(retry_delay)
