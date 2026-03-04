"""Tests for ws_feed.py Binance WebSocket kline feed."""
import pytest


class TestWsFeedCache:
    """WsFeed cache updates from parsed WebSocket messages."""

    def _make_kline_msg(self, symbol: str = "BTCUSDT", close: str = "51000.0", is_closed: bool = True) -> dict:
        return {
            "stream": f"{symbol.lower()}@kline_1h",
            "data": {
                "e": "kline",
                "k": {
                    "t": 1704067200000,
                    "o": "50000.0",
                    "h": "52000.0",
                    "l": "49000.0",
                    "c": close,
                    "v": "1234.5",
                    "x": is_closed,  # candle closed flag
                },
            },
        }

    def test_closed_candle_updates_cache(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=True))
        assert feed.get_latest_close("BTC/USDT") == pytest.approx(51_000.0)

    def test_open_candle_does_not_update_cache(self):
        """Non-closed candles are ignored — only use confirmed 1h bars."""
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=False))
        assert feed.get_latest_close("BTC/USDT") is None

    def test_unknown_symbol_returns_none(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        assert feed.get_latest_close("XYZ/USDT") is None

    def test_symbol_normalisation(self):
        """get_latest_close() accepts both 'BTC/USDT' and 'BTCUSDT' forms."""
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._handle(self._make_kline_msg("BTCUSDT", "51000.0", is_closed=True))
        assert feed.get_latest_close("BTC/USDT") == pytest.approx(51_000.0)
        assert feed.get_latest_close("BTCUSDT") == pytest.approx(51_000.0)

    def test_stop_sets_running_false(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT"])
        feed._running = True
        feed.stop()
        assert feed._running is False


class TestWsFeedReconnect:
    """Reconnect backoff doubles up to max."""

    def test_backoff_doubles(self):
        from ws_feed import _next_backoff
        assert _next_backoff(1.0) == pytest.approx(2.0)
        assert _next_backoff(32.0) == pytest.approx(60.0)  # capped at 60

    def test_stream_names(self):
        from ws_feed import WsFeed
        feed = WsFeed(["BTC/USDT", "ETH/USDT", "SOL/USDT"])
        names = feed._stream_names()
        assert "btcusdt@kline_1h" in names
        assert "ethusdt@kline_1h" in names
        assert "solusdt@kline_1h" in names
        assert len(names) == 3


class TestWsFeedRobustness:
    """Edge-case and robustness tests for WsFeed."""

    def _make_kline_msg(self, symbol: str = "BTCUSDT", is_closed: bool = True, include_c: bool = True) -> dict:
        kline: dict = {
            "t": 1704067200000,
            "o": "50000.0",
            "h": "52000.0",
            "l": "49000.0",
            "v": "1234.5",
            "x": is_closed,
        }
        if include_c:
            kline["c"] = "51000.0"
        return {
            "stream": f"{symbol.lower()}@kline_1h",
            "data": {"e": "kline", "k": kline},
        }

    def test_start_creates_daemon_thread(self):
        """start() creates a daemon thread named 'ws_feed'."""
        from unittest.mock import patch

        from ws_feed import WsFeed

        feed = WsFeed(["BTC/USDT"])
        with patch.object(feed, "_run_loop"):  # prevent actual asyncio loop from starting
            feed.start()
        assert feed._thread is not None
        assert feed._thread.daemon is True
        assert feed._thread.name == "ws_feed"

    def test_empty_symbols_raises(self):
        """WsFeed raises ValueError when constructed with empty symbols list."""
        from ws_feed import WsFeed

        with pytest.raises(ValueError, match="at least one symbol"):
            WsFeed([])

    def test_handle_missing_close_key_does_not_raise(self):
        """_handle() with a closed kline but missing 'c' key is silently skipped."""
        from ws_feed import WsFeed

        feed = WsFeed(["BTC/USDT"])
        msg = self._make_kline_msg(is_closed=True, include_c=False)
        feed._handle(msg)  # must not raise KeyError
        assert feed.get_latest_close("BTC/USDT") is None
