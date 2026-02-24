"""
Stock Index Correlation — S&P 500 / NASDAQ vs BTC.
Sources:
  - Primary: Yahoo Finance (yfinance library, free, no key needed)
  - Fallback: CCXT BTC/USDT data + requests-based SPY fetch from Yahoo

v7.0: Added CCXT fallback so correlation works even when yfinance proxy is blocked.

Note:
    yfinance is an optional soft dependency; correlation analysis degrades
    gracefully to the Yahoo Finance CSV fallback (pure ``requests``) if
    yfinance is not installed.  If both methods fail the provider returns a
    neutral signal with ``data.error = 'all_sources_failed'``.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import requests

from config import Config

_log = logging.getLogger(__name__)

# Browser-like headers for Yahoo Finance CSV endpoint
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/csv,application/json,*/*",
}


class CorrelationAnalyzer:
    """Tracks correlation between crypto and traditional markets.

    Computes the rolling Pearson correlation between BTC daily returns
    and SPY (S&P 500 ETF) daily returns over the past ~30 days, then
    maps that correlation to a bullish/bearish/neutral signal.

    Uses yfinance as the primary data source and falls back to direct
    Yahoo Finance CSV fetching when yfinance is unavailable or blocked.
    """

    def __init__(self) -> None:
        """Initialise the analyzer with an empty result cache."""
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._cache_ttl: int = 600  # 10 min — correlation doesn't change fast

    def get_signal(self) -> dict[str, Any]:
        """Compute rolling correlation between BTC and stock indices.

        Attempts yfinance first, then falls back to Yahoo Finance CSV,
        then returns a neutral signal if both fail.  Results are cached
        for ``_cache_ttl`` seconds.

        Returns:
            Dictionary with keys: ``source`` (``'correlation'``),
            ``signal`` (``'bullish'``, ``'bearish'``, or ``'neutral'``),
            ``strength`` (0.0–0.3), ``data`` containing
            ``btc_spy_correlation``, ``spy_trend``, ``window_days``,
            and ``method``.

        Note:
            Requires ``Config.ENABLE_CORRELATION`` to be ``True``; returns
            a neutral signal immediately when the feature is disabled.
            Equity correlations require yfinance to be installed.
            If yfinance is absent, only the CSV fallback is used.
        """
        if not Config.ENABLE_CORRELATION:
            return {"source": "correlation", "signal": "neutral", "strength": 0.0, "data": {}}

        # Cache check
        now = time.time()
        if now - self._cache_ts < self._cache_ttl and self._cache:
            return self._cache

        # Try yfinance first, then CCXT fallback
        result = self._try_yfinance()
        if result is None:
            result = self._try_yahoo_csv_fallback()
        if result is None:
            result = {
                "source": "correlation",
                "signal": "neutral",
                "strength": 0.0,
                "data": {"error": "all_sources_failed"},
            }

        self._cache = result
        self._cache_ts = now
        return result

    def _try_yfinance(self) -> dict[str, Any] | None:
        """Primary method: use yfinance library.

        Downloads BTC-USD and SPY daily OHLCV for the past month,
        computes daily returns, and calls ``_compute_signal()``.

        Returns:
            Correlation signal dict on success, or ``None`` if yfinance
            is not installed, returns insufficient data, or raises any
            exception.

        Note:
            yfinance is an optional dependency.  If it is not installed
            this method returns ``None`` immediately and the caller falls
            back to ``_try_yahoo_csv_fallback()``.
        """
        try:
            import yfinance as yf

            btc = yf.download("BTC-USD", period="1mo", interval="1d", progress=False)
            spy = yf.download("SPY", period="1mo", interval="1d", progress=False)

            if btc.empty or spy.empty or len(btc) < 10 or len(spy) < 10:
                _log.debug("yfinance returned insufficient data")
                return None

            btc_returns = btc["Close"].pct_change().dropna()
            spy_returns = spy["Close"].pct_change().dropna()

            # Handle multi-level columns from newer yfinance
            if hasattr(btc_returns, "columns"):
                btc_returns = btc_returns.iloc[:, 0] if len(btc_returns.shape) > 1 else btc_returns
            if hasattr(spy_returns, "columns"):
                spy_returns = spy_returns.iloc[:, 0] if len(spy_returns.shape) > 1 else spy_returns

            common = btc_returns.index.intersection(spy_returns.index)
            if len(common) < 5:
                return None

            return self._compute_signal(
                btc_returns.loc[common].values.flatten(),
                spy_returns.loc[common].values.flatten(),
                len(common),
                "yfinance",
            )
        except ImportError:
            _log.debug("yfinance not installed")
            return None
        except Exception as e:
            _log.debug("yfinance failed: %s", e)
            return None

    def _try_yahoo_csv_fallback(self) -> dict[str, Any] | None:
        """Fallback: fetch Yahoo Finance CSV directly via requests.

        Downloads ~35 days of daily close prices for BTC-USD and SPY
        using the Yahoo Finance v7 download endpoint, computes daily
        returns from the parsed CSV, and calls ``_compute_signal()``.

        Returns:
            Correlation signal dict on success, or ``None`` on failure.
        """
        try:
            now = int(time.time())
            period1 = now - 35 * 86400  # ~35 days ago

            btc_prices = self._fetch_yahoo_csv("BTC-USD", period1, now)
            spy_prices = self._fetch_yahoo_csv("SPY", period1, now)

            if btc_prices is None or spy_prices is None:
                return None
            if len(btc_prices) < 10 or len(spy_prices) < 10:
                return None

            # Compute returns (simple dict-based, keyed by date string)
            btc_returns = {}
            btc_dates = sorted(btc_prices.keys())
            for i in range(1, len(btc_dates)):
                prev, cur = btc_dates[i - 1], btc_dates[i]
                if btc_prices[prev] > 0:
                    btc_returns[cur] = (btc_prices[cur] - btc_prices[prev]) / btc_prices[prev]

            spy_returns = {}
            spy_dates = sorted(spy_prices.keys())
            for i in range(1, len(spy_dates)):
                prev, cur = spy_dates[i - 1], spy_dates[i]
                if spy_prices[prev] > 0:
                    spy_returns[cur] = (spy_prices[cur] - spy_prices[prev]) / spy_prices[prev]

            common = sorted(set(btc_returns.keys()) & set(spy_returns.keys()))
            if len(common) < 5:
                return None

            btc_arr = np.array([btc_returns[d] for d in common])
            spy_arr = np.array([spy_returns[d] for d in common])

            return self._compute_signal(btc_arr, spy_arr, len(common), "yahoo_csv")

        except Exception as e:
            _log.debug("Yahoo CSV fallback failed: %s", e)
            return None

    def _fetch_yahoo_csv(self, ticker: str, period1: int, period2: int) -> dict[str, float] | None:
        """Fetch daily close prices from Yahoo Finance CSV endpoint.

        Args:
            ticker: Yahoo Finance ticker symbol, e.g. ``'BTC-USD'`` or
                ``'SPY'``.
            period1: Start of the date range as a Unix timestamp (seconds).
            period2: End of the date range as a Unix timestamp (seconds).

        Returns:
            Dictionary mapping date strings (``'YYYY-MM-DD'``) to adjusted
            close prices, or ``None`` if the request fails or returns fewer
            than 5 data points.
        """
        url = (
            f"https://query1.finance.yahoo.com/v7/finance/download/{ticker}"
            f"?period1={period1}&period2={period2}&interval=1d&events=history"
        )
        try:
            resp = requests.get(url, headers=_HEADERS, timeout=8)
            if not resp.ok:
                return None

            prices = {}
            lines = resp.text.strip().split("\n")
            for line in lines[1:]:  # Skip header
                parts = line.split(",")
                if len(parts) >= 5:
                    date_str = parts[0]
                    try:
                        close = float(parts[4])  # Adj Close or Close
                        prices[date_str] = close
                    except (ValueError, IndexError):
                        continue
            return prices if len(prices) >= 5 else None
        except Exception as e:
            _log.debug("Yahoo CSV fetch failed for %s: %s", ticker, e)
            return None

    def _compute_signal(
        self, btc_returns: np.ndarray, spy_returns: np.ndarray, window: int, source_method: str
    ) -> dict[str, Any]:
        """Common correlation computation shared by both data-fetch methods.

        Args:
            btc_returns: Array of BTC daily return values.
            spy_returns: Array of SPY daily return values aligned to the
                same dates as ``btc_returns``.
            window: Number of overlapping data points used (reported in
                ``data.window_days``).
            source_method: Label for the data source used
                (``'yfinance'`` or ``'yahoo_csv'``).

        Returns:
            Standard intelligence provider signal dict with keys:
            ``source``, ``signal``, ``strength``, ``data``.  The ``data``
            sub-dict contains ``btc_spy_correlation``, ``spy_trend``,
            ``window_days``, and ``method``.
        """
        corr = np.corrcoef(btc_returns, spy_returns)[0, 1]
        spy_trend = "up" if spy_returns.mean() > 0 else "down"

        signal = "neutral"
        strength = 0.0
        if abs(corr) > 0.5:
            if spy_trend == "down":
                signal = "bearish"
                strength = abs(corr) * 0.3
            elif spy_trend == "up":
                signal = "bullish"
                strength = abs(corr) * 0.3

        return {
            "source": "correlation",
            "signal": signal,
            "strength": round(strength, 3),
            "data": {
                "btc_spy_correlation": round(float(corr), 4),
                "spy_trend": spy_trend,
                "window_days": window,
                "method": source_method,
            },
        }
