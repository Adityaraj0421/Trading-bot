"""SessionAnalyzer — maps UTC time to session confidence multiplier."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any


class SessionAnalyzer:
    """Maps current UTC time to a confidence multiplier.

    US session (peak liquidity) = 1.00 multiplier.
    EU session = 0.90. Asia = 0.75. Weekend = 0.60.
    When sessions overlap, the higher multiplier wins (US > EU > Asia).
    """

    def analyze(self, now: datetime | None = None) -> dict[str, Any]:
        """Return session label and confidence multiplier for the given time.

        Args:
            now: UTC datetime to evaluate. Defaults to datetime.now(UTC).

        Returns:
            Dict with keys ``session`` and ``confidence_multiplier``.
        """
        if now is None:
            now = datetime.now(UTC)

        if now.weekday() >= 5:  # Saturday=5, Sunday=6
            return {"session": "weekend", "confidence_multiplier": 0.60}

        hour = now.hour
        in_us = 13 <= hour < 22
        in_eu = 7 <= hour < 15
        in_asia = hour < 8

        if in_us:
            return {"session": "US", "confidence_multiplier": 1.00}
        if in_eu:
            return {"session": "EU", "confidence_multiplier": 0.90}
        if in_asia:
            return {"session": "Asia", "confidence_multiplier": 0.75}
        # Dead zone 22:00–00:00 UTC: thin liquidity, treat as Asia
        return {"session": "Asia", "confidence_multiplier": 0.75}
