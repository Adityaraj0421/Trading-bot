"""
Async Data Fetcher — Uses ccxt.async_support and aiohttp for non-blocking I/O.
Falls back to sync DataFetcher demo data if async exchange is unavailable.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import aiohttp
import pandas as pd

from config import Config

_log = logging.getLogger(__name__)


class AsyncDataFetcher:
    """
    Non-blocking data fetcher using ccxt async and aiohttp.
    Runs exchange OHLCV fetches and sentiment API calls concurrently.
    """

    def __init__(self) -> None:
        """Initialize the async fetcher with lazy exchange and session creation.

        The ccxt async exchange and aiohttp session are created on first use
        via ``_get_exchange()`` and ``_get_session()`` respectively.
        """
        self._exchange: Any = None
        self.using_demo: bool = False
        self._session: aiohttp.ClientSession | None = None

    async def _get_exchange(self) -> Any:
        """Lazy-init async exchange."""
        if self._exchange is None:
            try:
                import ccxt.async_support as ccxt_async

                exchange_class = getattr(ccxt_async, Config.EXCHANGE_ID)
                params = {"enableRateLimit": True}
                if not Config.is_paper_mode() and Config.API_KEY:
                    params["apiKey"] = Config.API_KEY
                    params["secret"] = Config.API_SECRET
                self._exchange = exchange_class(params)
            except Exception as e:
                _log.debug("Async exchange init failed: %s", e)
                self._exchange = None
        return self._exchange

    async def _get_session(self) -> aiohttp.ClientSession:
        """Lazy-init aiohttp session."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=5))
        return self._session

    async def fetch_ohlcv(
        self, symbol: str | None = None, timeframe: str | None = None, limit: int | None = None
    ) -> pd.DataFrame:
        """Fetch OHLCV candle data from the configured exchange asynchronously.

        Falls back to synthetic demo data (via ``demo_data.generate_ohlcv``)
        if the exchange is unavailable or returns an error. Sets
        ``self.using_demo`` accordingly.

        Args:
            symbol: Trading pair override (defaults to ``Config.TRADING_PAIR``).
            timeframe: Candle interval override (defaults to
                ``Config.TIMEFRAME``).
            limit: Number of candles to fetch (defaults to
                ``Config.LOOKBACK_BARS``).

        Returns:
            DataFrame indexed by timestamp with columns
            ``open``, ``high``, ``low``, ``close``, ``volume``.
        """
        symbol = symbol or Config.TRADING_PAIR
        timeframe = timeframe or Config.TIMEFRAME
        limit = limit or Config.LOOKBACK_BARS

        exchange = await self._get_exchange()
        if exchange:
            try:
                raw = await exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
                if raw:
                    df = pd.DataFrame(
                        raw,
                        columns=["timestamp", "open", "high", "low", "close", "volume"],
                    )
                    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
                    df.set_index("timestamp", inplace=True)
                    df = df[df["volume"] > 0]
                    self.using_demo = False
                    return df
            except Exception as e:
                if not self.using_demo:
                    _log.warning("[AsyncFetcher] Exchange error: %s", e)
                    _log.warning("[AsyncFetcher] Falling back to demo data")

        return self._generate_demo_data(limit, timeframe)

    async def fetch_fear_greed(self) -> dict:
        """Fetch the latest Fear & Greed index value from alternative.me.

        Returns:
            Dict with ``value`` (int, 0–100) and ``source``
            (``"async_api"`` on success, ``"fallback"`` on error).
        """
        url = "https://api.alternative.me/fng/?limit=7&format=json"
        try:
            session = await self._get_session()
            async with session.get(url) as resp:
                data = await resp.json()
                return {"value": int(data["data"][0]["value"]), "source": "async_api"}
        except Exception as e:
            _log.debug("Fear/Greed fetch failed: %s", e)
            return {"value": 50, "source": "fallback"}

    async def fetch_all(self) -> tuple[pd.DataFrame, dict[str, Any]]:
        """Fetch OHLCV and Fear & Greed index concurrently.

        Runs both API calls in parallel via ``asyncio.gather`` — the key
        performance optimization over sequential polling.

        Returns:
            Tuple of ``(ohlcv_dataframe, fear_greed_dict)``.
        """
        ohlcv_task = asyncio.create_task(self.fetch_ohlcv())
        fg_task = asyncio.create_task(self.fetch_fear_greed())

        df, fg_data = await asyncio.gather(ohlcv_task, fg_task)
        return df, fg_data

    def _generate_demo_data(self, periods: int = 200, timeframe: str = "1h") -> pd.DataFrame:
        """Sync fallback for demo data."""
        from demo_data import generate_ohlcv

        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 60)
        self.using_demo = True
        return generate_ohlcv(
            symbol=Config.TRADING_PAIR,
            periods=periods,
            timeframe_minutes=minutes,
        )

    async def close(self) -> None:
        """Close the ccxt exchange and aiohttp session to release resources."""
        if self._exchange:
            await self._exchange.close()
        if self._session and not self._session.closed:
            await self._session.close()
