"""OITrendAnalyzer — classify open interest trend vs price direction."""

from __future__ import annotations

from typing import Any

_MIN_CHANGE = 1.0  # % — below this is noise


class OITrendAnalyzer:
    """Classifies OI trend relative to price direction.

    Expanding OI + rising price → new longs being added (bullish conviction).
    Expanding OI + falling price → new shorts being added (bearish conviction).
    Contracting OI + rising price → short covering (weaker bullish signal).
    """

    def analyze(
        self, oi_change_pct: float | None, price_change_pct: float | None
    ) -> dict[str, Any]:
        """Classify OI vs price trend.

        Args:
            oi_change_pct: % change in open interest over lookback period.
            price_change_pct: % change in price over same period.

        Returns:
            Dict with key ``oi_trend``.
        """
        if oi_change_pct is None or price_change_pct is None:
            return {"oi_trend": "neutral"}
        if abs(oi_change_pct) < _MIN_CHANGE and abs(price_change_pct) < _MIN_CHANGE:
            return {"oi_trend": "neutral"}

        oi_up = oi_change_pct > _MIN_CHANGE
        oi_down = oi_change_pct < -_MIN_CHANGE
        price_up = price_change_pct > 0

        if oi_up and price_up:
            return {"oi_trend": "expanding_up"}
        if oi_down and not price_up:
            return {"oi_trend": "expanding_down"}
        if oi_down and price_up:
            return {"oi_trend": "contracting"}
        return {"oi_trend": "neutral"}
