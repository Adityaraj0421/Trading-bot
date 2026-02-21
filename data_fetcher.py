"""
Data fetching module using CCXT.
Connects to crypto exchanges and retrieves OHLCV + order book data.
Falls back to demo data if exchange is unreachable.
"""

import logging
import os
from pathlib import Path
import ccxt
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from config import Config

_log = logging.getLogger(__name__)


class DataFetcher:
    def __init__(self):
        self.exchange = self._init_exchange()
        self.using_demo = False

    def _init_exchange(self):
        """Initialize the exchange connection.

        Supports both HMAC (API_SECRET) and Ed25519 (API_PRIVATE_KEY_PATH)
        authentication methods.
        """
        exchange_class = getattr(ccxt, Config.EXCHANGE_ID)

        params = {"enableRateLimit": True}

        if Config.API_KEY:
            params["apiKey"] = Config.API_KEY

            # Ed25519 key authentication (recommended by Binance)
            if Config.API_PRIVATE_KEY_PATH:
                try:
                    # Validate path to prevent directory traversal attacks
                    key_path = Path(Config.API_PRIVATE_KEY_PATH).resolve()
                    allowed_dirs = [
                        Path.cwd().resolve(),
                        Path.home().resolve(),
                    ]
                    if not any(str(key_path).startswith(str(d)) for d in allowed_dirs):
                        raise ValueError(
                            f"Private key path '{key_path}' is outside allowed directories. "
                            f"Key must be in project dir or home dir."
                        )
                    # Warn if file permissions are too open (Unix only)
                    if hasattr(os, "stat"):
                        mode = os.stat(key_path).st_mode & 0o777
                        if mode & 0o077:  # Group or other can read
                            _log.warning(
                                "Private key file %s has permissive permissions (%o). "
                                "Consider: chmod 600 %s",
                                key_path, mode, key_path,
                            )
                    with open(key_path, "r") as f:
                        private_key = f.read().strip()
                    params["secret"] = private_key
                    # Tell CCXT this is an Ed25519 key for Binance
                    if Config.EXCHANGE_ID == "binance":
                        params["options"] = {"defaultType": "spot"}
                except FileNotFoundError:
                    _log.error(
                        "Ed25519 private key not found: %s", Config.API_PRIVATE_KEY_PATH
                    )
                except ValueError as e:
                    _log.error("Key path validation failed: %s", e)
            elif Config.API_SECRET:
                # Traditional HMAC secret
                params["secret"] = Config.API_SECRET

        exchange = exchange_class(params)

        return exchange

    def fetch_ohlcv(
        self, symbol: str = None, timeframe: str = None, limit: int = None
    ) -> pd.DataFrame:
        """
        Fetch OHLCV (candlestick) data from the exchange.
        Falls back to generated demo data if exchange is unreachable.

        Returns a DataFrame with columns:
            open, high, low, close, volume (indexed by timestamp)
        """
        symbol = symbol or Config.TRADING_PAIR
        timeframe = timeframe or Config.TIMEFRAME
        limit = limit or Config.LOOKBACK_BARS

        try:
            raw = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
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
        except (ccxt.NetworkError, ccxt.ExchangeError) as e:
            if not self.using_demo:
                _log.warning("[DataFetcher] Exchange unreachable: %s", e)
                _log.warning("[DataFetcher] Falling back to demo data")
        except Exception as e:
            if not self.using_demo:
                _log.warning("[DataFetcher] Unexpected error: %s", e)
                _log.warning("[DataFetcher] Falling back to demo data")

        # Fallback: generate demo data
        return self._generate_demo_data(limit, timeframe)

    def _generate_demo_data(
        self, periods: int = 200, timeframe: str = "1h"
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for offline testing."""
        from demo_data import generate_ohlcv

        tf_minutes = {"1m": 1, "5m": 5, "15m": 15, "1h": 60, "4h": 240, "1d": 1440}
        minutes = tf_minutes.get(timeframe, 60)

        self.using_demo = True
        return generate_ohlcv(
            symbol=Config.TRADING_PAIR,
            periods=periods,
            timeframe_minutes=minutes,
        )

    def fetch_ticker(self, symbol: str = None) -> dict:
        """Fetch current ticker (last price, bid, ask, volume)."""
        symbol = symbol or Config.TRADING_PAIR
        try:
            return self.exchange.fetch_ticker(symbol)
        except Exception as e:
            # Return simulated ticker from demo data
            df = self.fetch_ohlcv(limit=1)
            if not df.empty:
                price = df["close"].iloc[-1]
                return {
                    "last": price,
                    "bid": price * 0.9999,
                    "ask": price * 1.0001,
                    "baseVolume": df["volume"].iloc[-1],
                }
            return {}

    def fetch_order_book(self, symbol: str = None, depth: int = 10) -> dict:
        """Fetch order book for spread and liquidity analysis."""
        symbol = symbol or Config.TRADING_PAIR
        try:
            book = self.exchange.fetch_order_book(symbol, depth)
            return {
                "bid": book["bids"][0][0] if book["bids"] else None,
                "ask": book["asks"][0][0] if book["asks"] else None,
                "spread": (
                    (book["asks"][0][0] - book["bids"][0][0])
                    if book["bids"] and book["asks"]
                    else None
                ),
                "bid_volume": sum(b[1] for b in book["bids"]),
                "ask_volume": sum(a[1] for a in book["asks"]),
            }
        except Exception as e:
            _log.debug("Order book fetch failed: %s", e)
            return {}

    def get_available_pairs(self) -> list:
        """List available trading pairs on the exchange."""
        try:
            self.exchange.load_markets()
            return list(self.exchange.markets.keys())
        except Exception as e:
            _log.error("[DataFetcher] Error loading markets: %s", e)
            return []
