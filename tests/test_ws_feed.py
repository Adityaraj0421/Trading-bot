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
        assert len(names) == 3
