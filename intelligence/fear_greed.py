"""
Fear & Greed Index Intelligence Provider.

Fetches the Crypto Fear & Greed Index from alternative.me and returns a
contrarian signal: extreme fear is bullish (dip buying opportunity),
extreme greed is bearish (distribution zone).

API: https://api.alternative.me/fng/?limit=1
  Response: {"data": [{"value": "72", "value_classification": "Greed", ...}]}
"""

from __future__ import annotations

import logging
import time
from typing import Any

import requests

_log = logging.getLogger(__name__)

_FNG_URL = "https://api.alternative.me/fng/?limit=1"
_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; CryptoTradingBot/1.0)",
    "Accept": "application/json",
}


class FearGreedProvider:
    """Intelligence provider: Crypto Fear & Greed Index from alternative.me.

    Returns a contrarian signal — extreme fear is treated as a buy signal,
    extreme greed as a sell signal. Results are cached for ``_cache_ttl``
    seconds (3600s, since the index updates once daily).
    """

    def __init__(self) -> None:
        """Initialise with empty cache."""
        self._cache: dict[str, Any] = {}
        self._cache_ts: float = 0.0
        self._cache_ttl: int = 3600  # 1 hour — index updates once daily

    def get_signal(self) -> dict[str, Any]:
        """Fetch Fear & Greed Index and return contrarian signal.

        Returns:
            Dictionary with keys: ``source`` (``'fear_greed'``), ``signal``
            (``'bullish'``, ``'bearish'``, or ``'neutral'``), ``strength``
            (0.0–0.8), ``data`` containing ``value``, ``classification``,
            and ``strength``.
        """
        now = time.time()
        if self._cache and now - self._cache_ts < self._cache_ttl:
            return self._cache

        try:
            resp = requests.get(_FNG_URL, headers=_HEADERS, timeout=8)
            if resp.status_code == 429:
                _log.debug("FearGreed: rate limited")
                return self._neutral_signal("rate_limited")
            if resp.status_code != 200:
                return self._neutral_signal(f"http_{resp.status_code}")

            data = resp.json()
            entries = data.get("data", [])
            if not entries:
                return self._neutral_signal("empty_response")

            value = int(entries[0]["value"])
            classification = entries[0].get("value_classification", "")

            signal, strength = self._compute_signal(value)

            result: dict[str, Any] = {
                "source": "fear_greed",
                "signal": signal,
                "strength": round(strength, 3),
                "data": {
                    "value": value,
                    "classification": classification,
                    "strength": round(strength, 3),
                },
            }
            self._cache = result
            self._cache_ts = now
            return result

        except Exception as e:
            _log.debug("FearGreed fetch error: %s", e)
            return self._neutral_signal(str(e))

    def _compute_signal(self, value: int) -> tuple[str, float]:
        """Map a Fear & Greed value (0–100) to a contrarian signal + strength.

        Args:
            value: Fear & Greed Index value, 0 (extreme fear) to
                100 (extreme greed).

        Returns:
            Tuple of (signal, strength) where signal is ``'bullish'``,
            ``'bearish'``, or ``'neutral'``, and strength is in [0.0, 0.8].
        """
        distance_from_neutral = abs(value - 50)
        strength = min(distance_from_neutral / 50.0, 1.0) * 0.8

        if value <= 44:
            return "bullish", strength
        elif value <= 55:
            return "neutral", 0.0
        else:
            return "bearish", strength

    def _neutral_signal(self, reason: str) -> dict[str, Any]:
        """Return a neutral fallback signal on error or unavailability.

        Args:
            reason: Short description of why the signal is neutral.

        Returns:
            Dict with ``signal='neutral'``, ``strength=0.0``, and the
            reason in ``data``.
        """
        return {
            "source": "fear_greed",
            "signal": "neutral",
            "strength": 0.0,
            "data": {"error": reason},
        }
