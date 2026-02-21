"""
Real-time WebSocket Streamer
==============================
Replaces polling with persistent websocket connections to exchanges.
Supports Binance, Kraken, Coinbase — auto-detects from Config.EXCHANGE_ID.

Features:
  - Real-time trade/ticker/kline streams
  - Order book depth streaming
  - Automatic reconnection with exponential backoff
  - Thread-safe data access via callbacks
  - Heartbeat monitoring
"""

import json
import time
import logging
import asyncio
import threading
from collections import deque
from typing import Callable, Optional
from datetime import datetime

_log = logging.getLogger(__name__)

# Exchange WebSocket endpoints
WS_ENDPOINTS = {
    "binance": "wss://stream.binance.com:9443/ws",
    "kraken": "wss://ws.kraken.com",
    "coinbase": "wss://ws-feed.exchange.coinbase.com",
    "bybit": "wss://stream.bybit.com/v5/public/spot",
    "kucoin": None,  # KuCoin requires token fetch first
}


def _normalize_pair_binance(pair: str) -> str:
    """BTC/USDT -> btcusdt"""
    return pair.replace("/", "").lower()


def _normalize_pair_kraken(pair: str) -> str:
    """BTC/USDT -> XBT/USDT"""
    return pair.replace("BTC", "XBT")


class WebSocketStreamer:
    """
    Manages real-time websocket connections to crypto exchanges.
    Provides callbacks for trade, ticker, kline, and order book updates.
    """

    def __init__(self, exchange_id: str, trading_pair: str, timeframe: str = "1h"):
        self.exchange_id = exchange_id.lower()
        self.trading_pair = trading_pair
        self.timeframe = timeframe

        # Latest data (thread-safe via deques)
        self.latest_ticker: dict = {}
        self.latest_trades: deque = deque(maxlen=1000)
        self.latest_klines: deque = deque(maxlen=500)
        self.latest_orderbook: dict = {"bids": [], "asks": [], "ts": 0}

        # Callbacks
        self._on_ticker: Optional[Callable] = None
        self._on_trade: Optional[Callable] = None
        self._on_kline: Optional[Callable] = None
        self._on_orderbook: Optional[Callable] = None

        # Connection state
        self._running = False
        self._connected = False
        self._reconnect_count = 0
        self._max_reconnects = 50
        self._last_message_ts = 0
        self._heartbeat_timeout = 30  # seconds
        self._thread: Optional[threading.Thread] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None

        # Stats
        self.messages_received = 0
        self.connection_uptime_start = 0

    def on_ticker(self, callback: Callable):
        """Register callback for ticker updates: callback(ticker_dict)"""
        self._on_ticker = callback

    def on_trade(self, callback: Callable):
        """Register callback for trade updates: callback(trade_dict)"""
        self._on_trade = callback

    def on_kline(self, callback: Callable):
        """Register callback for kline/candle updates: callback(kline_dict)"""
        self._on_kline = callback

    def on_orderbook(self, callback: Callable):
        """Register callback for order book updates: callback(book_dict)"""
        self._on_orderbook = callback

    def start(self):
        """Start websocket connection in a background thread."""
        if self._running:
            _log.warning("WebSocket streamer already running")
            return

        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="ws-streamer")
        self._thread.start()
        _log.info("WebSocket streamer started for %s on %s", self.trading_pair, self.exchange_id)

    def stop(self):
        """Stop websocket connection."""
        self._running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5)
        self._connected = False
        _log.info("WebSocket streamer stopped")

    def is_connected(self) -> bool:
        return self._connected and self._running

    def get_status(self) -> dict:
        """Get streamer status."""
        uptime = time.time() - self.connection_uptime_start if self.connection_uptime_start else 0
        return {
            "connected": self._connected,
            "running": self._running,
            "exchange": self.exchange_id,
            "pair": self.trading_pair,
            "messages_received": self.messages_received,
            "reconnect_count": self._reconnect_count,
            "uptime_seconds": round(uptime),
            "last_message_age": round(time.time() - self._last_message_ts, 1) if self._last_message_ts else None,
        }

    def _run_loop(self):
        """Run the asyncio event loop in a thread."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._connect_with_retry())
        except Exception as e:
            _log.error("WebSocket loop error: %s", e)
        finally:
            self._loop.close()

    async def _connect_with_retry(self):
        """Connect with exponential backoff retry."""
        try:
            import websockets
        except ImportError:
            _log.error("websockets package not installed. pip install websockets")
            return

        while self._running and self._reconnect_count < self._max_reconnects:
            try:
                ws_url = self._build_ws_url()
                if not ws_url:
                    _log.error("No WebSocket URL for exchange: %s", self.exchange_id)
                    return

                _log.info("Connecting to %s...", ws_url)

                async with websockets.connect(ws_url, ping_interval=20, ping_timeout=10) as ws:
                    self._connected = True
                    self.connection_uptime_start = time.time()
                    self._reconnect_count = 0
                    _log.info("WebSocket connected to %s", self.exchange_id)

                    # Send subscription messages
                    await self._subscribe(ws)

                    # Start heartbeat monitor
                    heartbeat_task = asyncio.create_task(self._heartbeat_monitor(ws))

                    try:
                        async for message in ws:
                            if not self._running:
                                break
                            self._last_message_ts = time.time()
                            self.messages_received += 1
                            self._process_message(message)
                    finally:
                        heartbeat_task.cancel()

            except Exception as e:
                self._connected = False
                self._reconnect_count += 1
                backoff = min(2 ** self._reconnect_count, 60)
                _log.warning("WebSocket disconnected (%s), reconnecting in %ds... (%d/%d)",
                             e, backoff, self._reconnect_count, self._max_reconnects)
                await asyncio.sleep(backoff)

        if self._reconnect_count >= self._max_reconnects:
            _log.error("Max reconnection attempts reached. WebSocket streamer stopped.")
            self._running = False

    async def _heartbeat_monitor(self, ws):
        """Monitor connection health and force reconnect if stale."""
        while self._running:
            await asyncio.sleep(self._heartbeat_timeout)
            if self._last_message_ts and (time.time() - self._last_message_ts) > self._heartbeat_timeout:
                _log.warning("No messages in %ds, forcing reconnect", self._heartbeat_timeout)
                await ws.close()
                break

    def _build_ws_url(self) -> str:
        """Build exchange-specific WebSocket URL."""
        pair_lower = _normalize_pair_binance(self.trading_pair)
        tf_map = {"1m": "1m", "5m": "5m", "15m": "15m", "1h": "1h", "4h": "4h", "1d": "1d"}
        tf = tf_map.get(self.timeframe, "1h")

        if self.exchange_id == "binance":
            # Combined stream: ticker + klines + depth + trades
            streams = [
                f"{pair_lower}@ticker",
                f"{pair_lower}@kline_{tf}",
                f"{pair_lower}@depth20@100ms",
                f"{pair_lower}@trade",
            ]
            return f"wss://stream.binance.com:9443/stream?streams={'/'.join(streams)}"

        elif self.exchange_id == "kraken":
            return WS_ENDPOINTS["kraken"]

        elif self.exchange_id == "coinbase":
            return WS_ENDPOINTS["coinbase"]

        elif self.exchange_id == "bybit":
            return WS_ENDPOINTS["bybit"]

        return WS_ENDPOINTS.get(self.exchange_id, "")

    async def _subscribe(self, ws):
        """Send subscription messages for exchange-specific protocols."""
        if self.exchange_id == "kraken":
            pair = _normalize_pair_kraken(self.trading_pair)
            # Subscribe to ticker
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [pair],
                "subscription": {"name": "ticker"},
            }))
            # Subscribe to trades
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [pair],
                "subscription": {"name": "trade"},
            }))
            # Subscribe to book
            await ws.send(json.dumps({
                "event": "subscribe",
                "pair": [pair],
                "subscription": {"name": "book", "depth": 25},
            }))

        elif self.exchange_id == "coinbase":
            product_id = self.trading_pair.replace("/", "-")
            await ws.send(json.dumps({
                "type": "subscribe",
                "product_ids": [product_id],
                "channels": ["ticker", "level2_batch", "matches"],
            }))

        elif self.exchange_id == "bybit":
            pair_symbol = self.trading_pair.replace("/", "")
            await ws.send(json.dumps({
                "op": "subscribe",
                "args": [
                    f"tickers.{pair_symbol}",
                    f"publicTrade.{pair_symbol}",
                    f"orderbook.25.{pair_symbol}",
                ],
            }))

        # Binance: streams are in URL, no subscription needed

    def _process_message(self, raw_message: str):
        """Route and parse exchange-specific messages."""
        try:
            data = json.loads(raw_message)
        except json.JSONDecodeError:
            return

        if self.exchange_id == "binance":
            self._process_binance(data)
        elif self.exchange_id == "kraken":
            self._process_kraken(data)
        elif self.exchange_id == "coinbase":
            self._process_coinbase(data)
        elif self.exchange_id == "bybit":
            self._process_bybit(data)

    def _process_binance(self, data: dict):
        """Process Binance combined stream messages."""
        stream = data.get("stream", "")
        payload = data.get("data", data)

        if "@ticker" in stream:
            ticker = {
                "price": float(payload.get("c", 0)),
                "high_24h": float(payload.get("h", 0)),
                "low_24h": float(payload.get("l", 0)),
                "volume_24h": float(payload.get("v", 0)),
                "change_pct": float(payload.get("P", 0)),
                "ts": time.time(),
            }
            self.latest_ticker = ticker
            if self._on_ticker:
                self._on_ticker(ticker)

        elif "@trade" in stream:
            trade = {
                "price": float(payload.get("p", 0)),
                "quantity": float(payload.get("q", 0)),
                "side": "sell" if payload.get("m") else "buy",
                "ts": payload.get("T", 0) / 1000,
            }
            self.latest_trades.append(trade)
            if self._on_trade:
                self._on_trade(trade)

        elif "@kline" in stream:
            k = payload.get("k", {})
            kline = {
                "open": float(k.get("o", 0)),
                "high": float(k.get("h", 0)),
                "low": float(k.get("l", 0)),
                "close": float(k.get("c", 0)),
                "volume": float(k.get("v", 0)),
                "closed": k.get("x", False),
                "ts": k.get("t", 0) / 1000,
            }
            if kline["closed"]:
                self.latest_klines.append(kline)
            if self._on_kline:
                self._on_kline(kline)

        elif "@depth" in stream:
            book = {
                "bids": [[float(b[0]), float(b[1])] for b in payload.get("bids", [])],
                "asks": [[float(a[0]), float(a[1])] for a in payload.get("asks", [])],
                "ts": time.time(),
            }
            self.latest_orderbook = book
            if self._on_orderbook:
                self._on_orderbook(book)

    def _process_kraken(self, data):
        """Process Kraken messages."""
        if isinstance(data, dict):
            # System/subscription messages
            return
        if isinstance(data, list) and len(data) >= 4:
            channel = data[-2] if isinstance(data[-2], str) else ""
            payload = data[1]

            if "ticker" in channel:
                if isinstance(payload, dict):
                    ticker = {
                        "price": float(payload.get("c", [0])[0]),
                        "volume_24h": float(payload.get("v", [0, 0])[1]),
                        "high_24h": float(payload.get("h", [0, 0])[1]),
                        "low_24h": float(payload.get("l", [0, 0])[1]),
                        "ts": time.time(),
                    }
                    self.latest_ticker = ticker
                    if self._on_ticker:
                        self._on_ticker(ticker)

            elif "trade" in channel:
                if isinstance(payload, list):
                    for t in payload:
                        trade = {
                            "price": float(t[0]),
                            "quantity": float(t[1]),
                            "side": "buy" if t[3] == "b" else "sell",
                            "ts": float(t[2]),
                        }
                        self.latest_trades.append(trade)
                        if self._on_trade:
                            self._on_trade(trade)

    def _process_coinbase(self, data: dict):
        """Process Coinbase messages."""
        msg_type = data.get("type", "")

        if msg_type == "ticker":
            ticker = {
                "price": float(data.get("price", 0)),
                "volume_24h": float(data.get("volume_24h", 0)),
                "high_24h": float(data.get("high_24h", 0)),
                "low_24h": float(data.get("low_24h", 0)),
                "ts": time.time(),
            }
            self.latest_ticker = ticker
            if self._on_ticker:
                self._on_ticker(ticker)

        elif msg_type == "match" or msg_type == "last_match":
            trade = {
                "price": float(data.get("price", 0)),
                "quantity": float(data.get("size", 0)),
                "side": data.get("side", "buy"),
                "ts": time.time(),
            }
            self.latest_trades.append(trade)
            if self._on_trade:
                self._on_trade(trade)

    def _process_bybit(self, data: dict):
        """Process Bybit messages."""
        topic = data.get("topic", "")
        payload = data.get("data", {})

        if "tickers" in topic:
            ticker = {
                "price": float(payload.get("lastPrice", 0)),
                "volume_24h": float(payload.get("volume24h", 0)),
                "high_24h": float(payload.get("highPrice24h", 0)),
                "low_24h": float(payload.get("lowPrice24h", 0)),
                "change_pct": float(payload.get("price24hPcnt", 0)) * 100,
                "ts": time.time(),
            }
            self.latest_ticker = ticker
            if self._on_ticker:
                self._on_ticker(ticker)

        elif "publicTrade" in topic:
            if isinstance(payload, list):
                for t in payload:
                    trade = {
                        "price": float(t.get("p", 0)),
                        "quantity": float(t.get("v", 0)),
                        "side": t.get("S", "Buy").lower(),
                        "ts": t.get("T", 0) / 1000,
                    }
                    self.latest_trades.append(trade)
                    if self._on_trade:
                        self._on_trade(trade)

        elif "orderbook" in topic:
            book = {
                "bids": [[float(b[0]), float(b[1])] for b in payload.get("b", [])],
                "asks": [[float(a[0]), float(a[1])] for a in payload.get("a", [])],
                "ts": time.time(),
            }
            self.latest_orderbook = book
            if self._on_orderbook:
                self._on_orderbook(book)
