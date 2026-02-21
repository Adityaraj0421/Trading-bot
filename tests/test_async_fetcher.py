"""
Tests for AsyncDataFetcher — demo fallback, session lifecycle.
No real exchange connections: tests the fallback path and data shapes.
"""

import pytest
import asyncio
import pandas as pd
from async_fetcher import AsyncDataFetcher


@pytest.fixture()
def fetcher():
    return AsyncDataFetcher()


# ── Demo Fallback ─────────────────────────────────────────────────


class TestDemoFallback:
    def test_fetch_ohlcv_returns_dataframe(self, fetcher):
        """Without a real exchange, should fall back to demo data."""
        df = asyncio.run(fetcher.fetch_ohlcv(limit=100))
        assert isinstance(df, pd.DataFrame)
        assert len(df) > 0
        assert "close" in df.columns
        assert "volume" in df.columns

    def test_demo_data_method_sets_flag(self, fetcher):
        """_generate_demo_data sets using_demo flag."""
        fetcher._generate_demo_data(periods=50, timeframe="1h")
        assert fetcher.using_demo is True

    def test_fetch_fear_greed_fallback(self, fetcher):
        """Fear & Greed API should gracefully fallback."""
        result = asyncio.run(fetcher.fetch_fear_greed())
        assert "value" in result
        assert "source" in result
        assert isinstance(result["value"], int)

    def test_fetch_all_returns_tuple(self, fetcher):
        df, fg = asyncio.run(fetcher.fetch_all())
        assert isinstance(df, pd.DataFrame)
        assert isinstance(fg, dict)


# ── Session Lifecycle ─────────────────────────────────────────────


class TestSessionLifecycle:
    def test_initial_state(self, fetcher):
        assert fetcher._exchange is None
        assert fetcher._session is None
        assert fetcher.using_demo is False

    def test_close_without_init(self, fetcher):
        """close() should be safe even if nothing was initialized."""
        asyncio.run(fetcher.close())

    def test_close_after_fetch(self, fetcher):
        async def _do():
            await fetcher.fetch_ohlcv(limit=50)
            await fetcher.close()
        asyncio.run(_do())


# ── Demo Data Shape ───────────────────────────────────────────────


class TestDemoDataShape:
    def test_demo_data_columns(self, fetcher):
        df = fetcher._generate_demo_data(periods=100, timeframe="1h")
        assert set(["open", "high", "low", "close", "volume"]).issubset(df.columns)

    def test_demo_data_length(self, fetcher):
        df = fetcher._generate_demo_data(periods=200, timeframe="4h")
        assert len(df) == 200

    def test_demo_data_timeframe_mapping(self, fetcher):
        df = fetcher._generate_demo_data(periods=50, timeframe="5m")
        assert len(df) == 50
