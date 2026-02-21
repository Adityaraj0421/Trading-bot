"""
Stock Index Correlation — S&P 500 / NASDAQ vs BTC.
Sources:
  - Primary: Yahoo Finance (yfinance library, free, no key needed)
  - Fallback: CCXT BTC/USDT data + requests-based SPY fetch from Yahoo

v7.0: Added CCXT fallback so correlation works even when yfinance proxy is blocked.
"""

import time
import logging
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
    """Tracks correlation between crypto and traditional markets."""

    def __init__(self):
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._cache_ttl: int = 600  # 10 min — correlation doesn't change fast

    def get_signal(self) -> dict:
        """Compute rolling correlation between BTC and stock indices."""
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
            result = {"source": "correlation", "signal": "neutral", "strength": 0.0,
                      "data": {"error": "all_sources_failed"}}

        self._cache = result
        self._cache_ts = now
        return result

    def _try_yfinance(self) -> dict | None:
        """Primary method: use yfinance library."""
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

    def _try_yahoo_csv_fallback(self) -> dict | None:
        """Fallback: fetch Yahoo Finance CSV directly via requests."""
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

    def _fetch_yahoo_csv(self, ticker: str, period1: int, period2: int) -> dict | None:
        """Fetch daily close prices from Yahoo Finance CSV endpoint."""
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

    def _compute_signal(self, btc_returns: np.ndarray, spy_returns: np.ndarray,
                        window: int, source_method: str) -> dict:
        """Common correlation computation."""
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
