# tests/test_multi_timeframe_fetcher.py
from unittest.mock import MagicMock

import pytest

from data_snapshot import DataSnapshot
from multi_timeframe_fetcher import MultiTimeframeFetcher


def make_mock_exchange(fail_timeframe: str | None = None):
    """Return a mock CCXT exchange. If fail_timeframe is set, that timeframe raises."""
    ohlcv = [[1704067200000 + i * 3600000, 100.0, 101.0, 99.0, 100.5, 1000.0] for i in range(50)]
    exchange = MagicMock()

    def side_effect(symbol, timeframe, *args, **kwargs):
        if timeframe == fail_timeframe:
            raise Exception(f"rate limit on {timeframe}")
        return ohlcv

    exchange.fetch_ohlcv.side_effect = side_effect
    return exchange


class TestMultiTimeframeFetcher:
    def test_fetch_returns_datasnapshot(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        assert isinstance(snap, DataSnapshot)

    def test_fetch_sets_correct_symbol(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange())
        snap = fetcher.fetch("ETH/USDT")
        assert snap.symbol == "ETH/USDT"

    def test_fetch_populates_all_timeframes(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        assert snap.df_1h is not None
        assert snap.df_4h is not None
        assert snap.df_15m is not None

    def test_fetch_calls_exchange_three_times(self):
        exchange = make_mock_exchange()
        fetcher = MultiTimeframeFetcher(exchange=exchange)
        fetcher.fetch("BTC/USDT")
        assert exchange.fetch_ohlcv.call_count == 3

    def test_partial_failure_graceful_degradation(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange(fail_timeframe="4h"))
        snap = fetcher.fetch("BTC/USDT")
        assert snap.df_1h is not None
        assert snap.df_4h is None   # failed gracefully
        assert snap.df_15m is not None

    def test_dataframes_have_ohlcv_columns(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        for col in ("open", "high", "low", "close", "volume"):
            assert col in snap.df_1h.columns

    def test_dataframes_are_readonly(self):
        fetcher = MultiTimeframeFetcher(exchange=make_mock_exchange())
        snap = fetcher.fetch("BTC/USDT")
        with pytest.raises(ValueError):
            snap.df_1h.iloc[0, 0] = 999.0
