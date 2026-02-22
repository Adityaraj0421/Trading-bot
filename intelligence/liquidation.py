"""
Liquidation Intelligence Provider v1.0
========================================
Estimates liquidation clusters using open interest + funding rate data.
Since real-time liquidation data requires paid APIs (CoinGlass/Coinalyze),
this provider estimates clusters using:
  - Binance long/short ratio
  - Price levels where concentrated liquidations are likely
  - Large OI changes at specific price levels

Signal logic:
  - Large liquidation cluster BELOW price → magnetic pull down (bearish)
  - Large liquidation cluster ABOVE price → magnetic pull up (bullish)
  - Recent mass liquidation event → trend continuation after flush
"""

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

# Binance endpoints
_LONG_SHORT_URL = "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"
_TOP_LONG_SHORT_URL = "https://fapi.binance.com/futures/data/topLongShortAccountRatio"
_TAKER_BUY_SELL_URL = "https://fapi.binance.com/futures/data/takerlongshortRatio"


class LiquidationAnalyzer:
    """Intelligence provider: liquidation cluster estimation."""

    def __init__(self, symbols: list[str] | None = None) -> None:
        self.symbols = symbols or ["BTCUSDT", "ETHUSDT", "SOLUSDT"]
        self._cache: dict = {}
        self._cache_ts: float = 0
        self._cache_ttl: int = 300
        self._ls_history: dict[str, deque] = {s: deque(maxlen=24) for s in self.symbols}

    def get_signal(self) -> dict[str, Any]:
        """Return aggregated liquidation risk signal."""
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
                _log.debug("Liquidation error for %s: %s", symbol, e)
                per_symbol[symbol] = {"error": str(e)}

        if n == 0:
            result = {
                "source": "liquidation",
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

        result = {
            "source": "liquidation",
            "signal": signal,
            "strength": round(min(abs(net), 1.0), 3),
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
        """Analyze liquidation risk for a single symbol."""
        bull = 0.0
        bear = 0.0
        signals = []

        # 1. Global long/short ratio
        ls_ratio = self._fetch_long_short_ratio(symbol)
        if ls_ratio is not None:
            self._ls_history[symbol].append(ls_ratio)

            # Extreme long/short imbalance → contrarian signal
            if ls_ratio > 2.0:
                # Way more longs → liquidation cascade risk if price drops
                bear += 0.35
                signals.append(f"extreme_long_ratio={ls_ratio:.2f}")
            elif ls_ratio > 1.5:
                bear += 0.15
                signals.append(f"high_long_ratio={ls_ratio:.2f}")
            elif ls_ratio < 0.5:
                # Way more shorts → short squeeze risk if price rises
                bull += 0.35
                signals.append(f"extreme_short_ratio={ls_ratio:.2f}")
            elif ls_ratio < 0.67:
                bull += 0.15
                signals.append(f"high_short_ratio={ls_ratio:.2f}")

            # Trend in L/S ratio
            history = list(self._ls_history[symbol])
            if len(history) >= 4:
                recent = sum(history[-2:]) / 2
                older = sum(history[:2]) / 2
                if recent > older * 1.2:
                    bear += 0.1  # Longs increasing rapidly
                    signals.append("longs_increasing")
                elif recent < older * 0.8:
                    bull += 0.1  # Shorts increasing rapidly
                    signals.append("shorts_increasing")

        # 2. Top trader long/short ratio (whale positioning)
        top_ls = self._fetch_top_trader_ls(symbol)
        if top_ls is not None:
            if top_ls > 2.5:
                bear += 0.2  # Top traders heavily long → potential trap
                signals.append(f"top_traders_long={top_ls:.2f}")
            elif top_ls < 0.4:
                bull += 0.2  # Top traders heavily short → potential short trap
                signals.append(f"top_traders_short={top_ls:.2f}")

            # Divergence: if top traders vs retail disagree
            if ls_ratio and top_ls:
                if ls_ratio > 1.5 and top_ls < 0.8:
                    bull += 0.15  # Retail long, whales short → whales often right
                    signals.append("whale_retail_divergence_bearish_retail")
                elif ls_ratio < 0.7 and top_ls > 1.3:
                    bear += 0.15
                    signals.append("whale_retail_divergence_bullish_retail")

        # 3. Taker buy/sell volume ratio (aggression)
        taker_ratio = self._fetch_taker_buy_sell(symbol)
        if taker_ratio is not None:
            if taker_ratio > 1.3:
                bull += 0.15  # Aggressive buying
                signals.append(f"aggressive_buying={taker_ratio:.2f}")
            elif taker_ratio < 0.7:
                bear += 0.15  # Aggressive selling
                signals.append(f"aggressive_selling={taker_ratio:.2f}")

        return {
            "long_short_ratio": ls_ratio,
            "top_trader_ls": top_ls,
            "taker_ratio": taker_ratio,
            "bullish_score": round(bull, 3),
            "bearish_score": round(bear, 3),
            "signals": signals,
        }

    def _fetch_long_short_ratio(self, symbol: str) -> float | None:
        """Fetch global long/short account ratio."""
        try:
            resp = requests.get(
                _LONG_SHORT_URL,
                params={"symbol": symbol, "period": "1h", "limit": 1},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data:
                return float(data[0]["longShortRatio"])
        except Exception as e:
            _log.debug("L/S ratio fetch error: %s", e)
        return None

    def _fetch_top_trader_ls(self, symbol: str) -> float | None:
        """Fetch top trader long/short ratio."""
        try:
            resp = requests.get(
                _TOP_LONG_SHORT_URL,
                params={"symbol": symbol, "period": "1h", "limit": 1},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data:
                return float(data[0]["longShortRatio"])
        except Exception as e:
            _log.debug("Top trader L/S fetch error: %s", e)
        return None

    def _fetch_taker_buy_sell(self, symbol: str) -> float | None:
        """Fetch taker buy/sell volume ratio."""
        try:
            resp = requests.get(
                _TAKER_BUY_SELL_URL,
                params={"symbol": symbol, "period": "1h", "limit": 1},
                headers=_HEADERS,
                timeout=8,
            )
            if resp.status_code != 200:
                return None
            data = resp.json()
            if data:
                return float(data[0]["buySellRatio"])
        except Exception as e:
            _log.debug("Taker buy/sell fetch error: %s", e)
        return None
