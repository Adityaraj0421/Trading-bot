"""
Tests for WebSocketStreamer — message parsing and status tracking.
No real WebSocket connections: tests exercise parsing, URL building,
and state management directly.
"""

import json
import pytest
from websocket_streamer import (
    WebSocketStreamer, _normalize_pair_binance, _normalize_pair_kraken,
)


@pytest.fixture()
def streamer():
    return WebSocketStreamer("binance", "BTC/USDT", "1h")


# ── Pair Normalization ────────────────────────────────────────────


class TestPairNormalization:
    def test_binance_normalize(self):
        assert _normalize_pair_binance("BTC/USDT") == "btcusdt"

    def test_binance_normalize_no_slash(self):
        assert _normalize_pair_binance("ETHUSDT") == "ethusdt"

    def test_kraken_normalize(self):
        assert _normalize_pair_kraken("BTC/USDT") == "XBT/USDT"

    def test_kraken_normalize_no_btc(self):
        assert _normalize_pair_kraken("ETH/USDT") == "ETH/USDT"


# ── URL Building ──────────────────────────────────────────────────


class TestURLBuilding:
    def test_binance_url(self):
        s = WebSocketStreamer("binance", "BTC/USDT", "1h")
        url = s._build_ws_url()
        assert "stream.binance.com" in url
        assert "btcusdt@ticker" in url
        assert "btcusdt@kline_1h" in url

    def test_kraken_url(self):
        s = WebSocketStreamer("kraken", "BTC/USDT")
        url = s._build_ws_url()
        assert "ws.kraken.com" in url

    def test_coinbase_url(self):
        s = WebSocketStreamer("coinbase", "BTC/USDT")
        url = s._build_ws_url()
        assert "ws-feed.exchange.coinbase.com" in url

    def test_bybit_url(self):
        s = WebSocketStreamer("bybit", "BTC/USDT")
        url = s._build_ws_url()
        assert "stream.bybit.com" in url

    def test_unknown_exchange_empty(self):
        s = WebSocketStreamer("unknown_exchange", "BTC/USDT")
        assert s._build_ws_url() == ""


# ── Status Tracking ───────────────────────────────────────────────


class TestStatus:
    def test_initial_status(self, streamer):
        status = streamer.get_status()
        assert status["connected"] is False
        assert status["running"] is False
        assert status["exchange"] == "binance"
        assert status["pair"] == "BTC/USDT"
        assert status["messages_received"] == 0

    def test_is_connected_initial(self, streamer):
        assert streamer.is_connected() is False

    def test_callback_registration(self, streamer):
        called = []
        streamer.on_ticker(lambda t: called.append(t))
        assert streamer._on_ticker is not None


# ── Binance Message Parsing ───────────────────────────────────────


class TestBinanceParsing:
    def test_parse_ticker(self, streamer):
        msg = json.dumps({
            "stream": "btcusdt@ticker",
            "data": {"c": "50123.45", "h": "51000", "l": "49000",
                     "v": "1234.5", "P": "2.5"},
        })
        received = []
        streamer.on_ticker(lambda t: received.append(t))
        streamer._process_message(msg)
        assert len(received) == 1
        assert received[0]["price"] == 50123.45
        assert received[0]["volume_24h"] == 1234.5

    def test_parse_trade(self, streamer):
        msg = json.dumps({
            "stream": "btcusdt@trade",
            "data": {"p": "50100", "q": "0.5", "m": True, "T": 1700000000000},
        })
        received = []
        streamer.on_trade(lambda t: received.append(t))
        streamer._process_message(msg)
        assert len(received) == 1
        assert received[0]["price"] == 50100.0
        assert received[0]["side"] == "sell"  # m=True means maker was buy → taker was sell

    def test_parse_kline_closed(self, streamer):
        msg = json.dumps({
            "stream": "btcusdt@kline_1h",
            "data": {"k": {"o": "50000", "h": "51000", "l": "49000",
                           "c": "50500", "v": "100", "x": True, "t": 1700000000000}},
        })
        received = []
        streamer.on_kline(lambda k: received.append(k))
        streamer._process_message(msg)
        assert len(received) == 1
        assert received[0]["closed"] is True
        assert len(streamer.latest_klines) == 1

    def test_parse_kline_not_closed_not_appended(self, streamer):
        msg = json.dumps({
            "stream": "btcusdt@kline_1h",
            "data": {"k": {"o": "50000", "h": "51000", "l": "49000",
                           "c": "50500", "v": "100", "x": False, "t": 1700000000000}},
        })
        streamer._process_message(msg)
        assert len(streamer.latest_klines) == 0

    def test_parse_depth(self, streamer):
        msg = json.dumps({
            "stream": "btcusdt@depth20@100ms",
            "data": {"bids": [["50000", "1.0"], ["49999", "2.0"]],
                     "asks": [["50001", "0.5"]]},
        })
        received = []
        streamer.on_orderbook(lambda b: received.append(b))
        streamer._process_message(msg)
        assert len(received) == 1
        assert received[0]["bids"][0] == [50000.0, 1.0]

    def test_invalid_json_ignored(self, streamer):
        streamer._process_message("not json at all")
        assert streamer.messages_received == 0


# ── Kraken Message Parsing ────────────────────────────────────────


class TestKrakenParsing:
    def test_parse_kraken_ticker(self):
        s = WebSocketStreamer("kraken", "BTC/USDT")
        data = [0, {"c": ["50000.0", "1"], "v": ["100", "200"],
                     "h": ["51000", "51000"], "l": ["49000", "49000"]},
                "ticker", "XBT/USDT"]
        received = []
        s.on_ticker(lambda t: received.append(t))
        s._process_kraken(data)
        assert len(received) == 1
        assert received[0]["price"] == 50000.0

    def test_parse_kraken_trade(self):
        s = WebSocketStreamer("kraken", "BTC/USDT")
        data = [0, [["50000.0", "0.5", "1700000000.0", "b", "m", ""]],
                "trade", "XBT/USDT"]
        received = []
        s.on_trade(lambda t: received.append(t))
        s._process_kraken(data)
        assert len(received) == 1
        assert received[0]["side"] == "buy"

    def test_kraken_system_message_ignored(self):
        s = WebSocketStreamer("kraken", "BTC/USDT")
        s._process_kraken({"event": "systemStatus", "status": "online"})
        # No crash, no data stored


# ── Coinbase Message Parsing ──────────────────────────────────────


class TestCoinbaseParsing:
    def test_parse_coinbase_ticker(self):
        s = WebSocketStreamer("coinbase", "BTC/USDT")
        data = {"type": "ticker", "price": "50000.0", "volume_24h": "1000",
                "high_24h": "51000", "low_24h": "49000"}
        received = []
        s.on_ticker(lambda t: received.append(t))
        s._process_coinbase(data)
        assert len(received) == 1
        assert received[0]["price"] == 50000.0

    def test_parse_coinbase_match(self):
        s = WebSocketStreamer("coinbase", "BTC/USDT")
        data = {"type": "match", "price": "50100", "size": "0.3", "side": "buy"}
        received = []
        s.on_trade(lambda t: received.append(t))
        s._process_coinbase(data)
        assert len(received) == 1
        assert received[0]["quantity"] == 0.3


# ── Bybit Message Parsing ────────────────────────────────────────


class TestBybitParsing:
    def test_parse_bybit_ticker(self):
        s = WebSocketStreamer("bybit", "BTC/USDT")
        data = {"topic": "tickers.BTCUSDT",
                "data": {"lastPrice": "50000", "volume24h": "1000",
                         "highPrice24h": "51000", "lowPrice24h": "49000",
                         "price24hPcnt": "0.025"}}
        received = []
        s.on_ticker(lambda t: received.append(t))
        s._process_bybit(data)
        assert len(received) == 1
        assert received[0]["price"] == 50000.0
        assert received[0]["change_pct"] == 2.5

    def test_parse_bybit_trade(self):
        s = WebSocketStreamer("bybit", "BTC/USDT")
        data = {"topic": "publicTrade.BTCUSDT",
                "data": [{"p": "50100", "v": "0.5", "S": "Buy", "T": 1700000000000}]}
        received = []
        s.on_trade(lambda t: received.append(t))
        s._process_bybit(data)
        assert len(received) == 1
        assert received[0]["side"] == "buy"


# ── Lifecycle Guards ──────────────────────────────────────────────


class TestLifecycle:
    def test_stop_without_start(self, streamer):
        """Calling stop() before start() should not raise."""
        streamer.stop()
        assert streamer._running is False

    def test_start_sets_running(self, streamer):
        """start() sets running flag (thread will fail on websocket connect,
        but the flag should be set)."""
        streamer.start()
        assert streamer._running is True
        streamer.stop()
