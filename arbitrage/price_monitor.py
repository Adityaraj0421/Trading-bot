"""
Price Monitor — concurrent price fetching from multiple exchanges.
Uses CCXT for unified exchange access.
"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from config import Config


class PriceMonitor:
    """Fetches current prices from multiple exchanges concurrently."""

    def __init__(self, exchanges: dict = None):
        """
        Args:
            exchanges: dict of {name: ccxt.Exchange} instances.
                       If None, creates paper-mode instances.
        """
        self.exchanges = exchanges or {}
        self._last_prices = {}
        self._last_fetch_time = {}

    def fetch_all_prices(self, pair: str = None) -> dict:
        """
        Fetch current price for a pair from all exchanges concurrently.
        Returns: {exchange_name: {"bid": float, "ask": float, "last": float, "timestamp": float}}
        """
        pair = pair or Config.TRADING_PAIR
        results = {}

        with ThreadPoolExecutor(max_workers=len(self.exchanges) or 1) as executor:
            futures = {}
            for name, exchange in self.exchanges.items():
                futures[executor.submit(self._fetch_ticker, exchange, pair)] = name

            for future in as_completed(futures, timeout=10):
                name = futures[future]
                try:
                    ticker = future.result()
                    results[name] = ticker
                    self._last_prices[name] = ticker
                    self._last_fetch_time[name] = time.time()
                except Exception as e:
                    results[name] = {"error": str(e)}

        return results

    def _fetch_ticker(self, exchange, pair: str) -> dict:
        """Fetch ticker from a single exchange."""
        try:
            ticker = exchange.fetch_ticker(pair)
            return {
                "bid": ticker.get("bid", 0) or 0,
                "ask": ticker.get("ask", 0) or 0,
                "last": ticker.get("last", 0) or 0,
                "timestamp": ticker.get("timestamp", 0),
                "volume_24h": ticker.get("quoteVolume", 0) or 0,
            }
        except Exception as e:
            return {"error": str(e)}

    def get_last_prices(self) -> dict:
        return self._last_prices.copy()
