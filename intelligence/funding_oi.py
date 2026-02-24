"""
Funding Rate + Open Interest Intelligence Provider v1.0
=========================================================
Fetches perpetual futures funding rates and open interest from Binance.
Generates signals based on:
  - Extreme funding rates (overleveraged positioning)
  - OI spikes without price movement (impending volatility)
  - Funding rate trend reversal (smart money repositioning)

Signal logic:
  - Funding > +0.05% = overleveraged longs → bearish bias
  - Funding < -0.05% = overleveraged shorts → bullish bias
  - OI spike (>15% in 8h) + flat price = volatility warning (neutral but high strength)
  - Negative funding + rising OI = smart accumulation → bullish
  - Positive funding + falling OI = distribution → bearish
"""

from __future__ import annotations

import logging
import time
from collections import deque
from typing import Any

import requests

_log = logging.getLogger(__name__)

_HEADERS = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json",
}

# Binance Futures API endpoints
_FUNDING_URL = "https://fapi.binance.com/fapi/v1/fundingRate"
_OI_URL = "https://fapi.binance.com/fapi/v1/openInterest"
_OI_HIST_URL = "https://fapi.binance.com/futures/data/openInterestHist"
_MARK_PRICE_URL = "https://fapi.binance.com/fapi/v1/premiumIndex"


class FundingOIAnalyzer:
    """Intelligence provider: Binance perpetual futures funding rate + open interest.

    Aggregates funding rate and open interest signals across multiple symbols
    and returns a composite bullish/bearish/neutral signal.  Results are
    cached for ``_cache_ttl`` seconds.
    """

    def __init__(self, symbols: list[str] | None = None) -> None:
        """Initialise the analyzer for a set of Binance futures symbols.

        Args:
            symbols: List of Binance futures symbol strings
                (e.g. ``['BTCUSDT', 'ETHUSDT', 'SOLUSDT']``).
                Defaults to ``['BTCUSDT', 'ETHUSDT', 'SOLUSDT']`` when
                ``None``.
        """
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._cache_ttl: int = 300  # 5 min cache (funding updates every 8h)
        self._funding_history: dict[str, deque] = {s: deque(maxlen=24) for s in self.symbols}
        self._oi_history: dict[str, deque] = {s: deque(maxlen=24) for s in self.symbols}

    @staticmethod
    def _pair_to_binance(pair: str) -> str:
        """Convert 'BTC/USDT' to 'BTCUSDT'.

        Args:
            pair: Trading pair string with separator, e.g. ``'BTC/USDT'``
                or ``'BTC-USDT'``.

        Returns:
            Binance-format symbol string with separator removed.
        """
        return pair.replace("/", "").replace("-", "")

    def get_signal(self) -> dict[str, Any]:
        """Return aggregated funding/OI signal across all tracked symbols.

        Computes per-symbol bullish and bearish scores, averages them, and
        returns a composite signal.  Results are cached for ``_cache_ttl``
        seconds.

        Returns:
            Dictionary with keys: ``source`` (``'funding_oi'``), ``signal``
            (``'bullish'``, ``'bearish'``, or ``'neutral'``), ``strength``
            (0.0–1.0), ``data`` containing ``bullish_score``,
            ``bearish_score``, ``net_score``, and ``per_symbol`` details.
        """
        now = time.time()
        if self._cache and now - self._cache_ts < self._cache_ttl:
            return self._cache

        per_symbol = {}
        total_bull = 0.0
        total_bear = 0.0
        n = 0

        for symbol in self.symbols:
            try:
                data = self._analyze_symbol(symbol)
                per_symbol[symbol] = data
                total_bull += data.get("bullish_score", 0)
                total_bear += data.get("bearish_score", 0)
                n += 1
            except Exception as e:
                _log.debug("FundingOI error for %s: %s", symbol, e)
                per_symbol[symbol] = {"error": str(e)}

        if n == 0:
            result = {
                "source": "funding_oi",
                "signal": "neutral",
                "strength": 0.0,
                "data": {"error": "all_symbols_failed"},
            }
            self._cache = result
            self._cache_ts = now
            return result

        avg_bull = total_bull / n
        avg_bear = total_bear / n
        net = avg_bull - avg_bear

        if net > 0.15:
            signal = "bullish"
        elif net < -0.15:
            signal = "bearish"
        else:
            signal = "neutral"

        strength = min(abs(net), 1.0)

        result = {
            "source": "funding_oi",
            "signal": signal,
            "strength": round(strength, 3),
            "data": {
                "bullish_score": round(avg_bull, 3),
                "bearish_score": round(avg_bear, 3),
                "net_score": round(net, 3),
                "per_symbol": per_symbol,
            },
        }
        self._cache = result
        self._cache_ts = now
        return result

    def _analyze_symbol(self, symbol: str) -> dict[str, Any]:
        """Analyze funding rate + OI for a single symbol.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Dictionary with keys: ``funding_rate``, ``open_interest``,
            ``mark_price``, ``bullish_score``, ``bearish_score``,
            ``signals`` (list of human-readable signal descriptions).
        """
        funding = self._fetch_funding(symbol)
        oi = self._fetch_open_interest(symbol)
        mark = self._fetch_mark_price(symbol)

        bull = 0.0
        bear = 0.0
        signals = []

        # --- Funding rate analysis ---
        if funding is not None:
            self._funding_history[symbol].append(funding)

            # Extreme funding: contrarian signal
            if funding > 0.0005:  # >0.05% = heavily long
                bear += 0.4
                signals.append(f"high_funding={funding:.4%}")
            elif funding > 0.0003:
                bear += 0.15
                signals.append(f"elevated_funding={funding:.4%}")
            elif funding < -0.0005:  # <-0.05% = heavily short
                bull += 0.4
                signals.append(f"negative_funding={funding:.4%}")
            elif funding < -0.0003:
                bull += 0.15
                signals.append(f"low_funding={funding:.4%}")

            # Funding trend (last 3 readings)
            history = list(self._funding_history[symbol])
            if len(history) >= 3:
                recent_avg = sum(history[-3:]) / 3
                older_avg = sum(history[:3]) / 3 if len(history) >= 6 else recent_avg
                if recent_avg > older_avg + 0.0002:
                    bear += 0.1  # Funding rising = more longs piling in
                    signals.append("funding_trend_rising")
                elif recent_avg < older_avg - 0.0002:
                    bull += 0.1  # Funding falling = shorts building
                    signals.append("funding_trend_falling")

        # --- Open Interest analysis ---
        if oi is not None and oi > 0:
            self._oi_history[symbol].append(oi)
            history = list(self._oi_history[symbol])

            if len(history) >= 3:
                oi_change = (history[-1] - history[-3]) / history[-3]

                # OI spike detection (>15% increase in recent readings)
                if oi_change > 0.15:
                    # OI spike = big move coming, direction depends on funding
                    if funding and funding > 0.0002:
                        bear += 0.25  # Longs piling in + OI spike = long squeeze risk
                        signals.append(f"oi_spike_long_risk={oi_change:.1%}")
                    elif funding and funding < -0.0002:
                        bull += 0.25  # Shorts piling in + OI spike = short squeeze risk
                        signals.append(f"oi_spike_short_squeeze={oi_change:.1%}")
                    else:
                        signals.append(f"oi_spike_neutral={oi_change:.1%}")

                # OI decline = positions closing = trend exhaustion
                elif oi_change < -0.10:
                    signals.append(f"oi_declining={oi_change:.1%}")

        # --- Composite: funding + OI divergence ---
        if funding and oi and len(self._oi_history[symbol]) >= 3:
            oi_history = list(self._oi_history[symbol])
            oi_rising = oi_history[-1] > oi_history[0] * 1.05 if len(oi_history) > 1 else False
            if funding < -0.0002 and oi_rising:
                bull += 0.2  # Smart accumulation: negative funding + rising OI
                signals.append("smart_accumulation")
            elif funding > 0.0003 and not oi_rising:
                bear += 0.15  # Distribution: positive funding + flat/falling OI
                signals.append("distribution")

        return {
            "funding_rate": funding,
            "open_interest": oi,
            "mark_price": mark,
            "bullish_score": round(bull, 3),
            "bearish_score": round(bear, 3),
            "signals": signals,
        }

    def _fetch_funding(self, symbol: str) -> float | None:
        """Fetch latest funding rate from Binance.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Latest funding rate as a float, or ``None`` on failure or
            rate-limiting.
        """
        try:
            resp = requests.get(
                _FUNDING_URL,
                params={"symbol": symbol, "limit": 1},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code == 429:
                _log.debug("Rate limited: funding rate")
                return None
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data and len(data) > 0:
                return float(data[0]["fundingRate"])
        except Exception as e:
            _log.debug("Funding rate fetch error: %s", e)
        return None

    def _fetch_open_interest(self, symbol: str) -> float | None:
        """Fetch current open interest from Binance.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Current open interest as a float (in base asset units), or
            ``None`` on failure or rate-limiting.
        """
        try:
            resp = requests.get(
                _OI_URL,
                params={"symbol": symbol},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code == 429:
                return None
            if resp.status_code != 200:
                return None
            data = resp.json()
            return float(data.get("openInterest", 0))
        except Exception as e:
            _log.debug("OI fetch error: %s", e)
        return None

    def _fetch_mark_price(self, symbol: str) -> float | None:
        """Fetch mark price + funding info from premiumIndex.

        Args:
            symbol: Binance futures symbol string, e.g. ``'BTCUSDT'``.

        Returns:
            Current mark price as a float, or ``None`` on failure.
        """
        try:
            resp = requests.get(
                _MARK_PRICE_URL,
                params={"symbol": symbol},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            return float(data.get("markPrice", 0))
        except Exception as e:
            _log.debug("Mark price fetch error: %s", e)
        return None
