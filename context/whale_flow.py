"""WhaleFlowAnalyzer — classify net whale exchange flow direction."""

from __future__ import annotations

from typing import Any

_THRESHOLD = 100_000.0  # USD notional — below this is noise


class WhaleFlowAnalyzer:
    """Classifies net whale flow into accumulating / distributing / neutral.

    Outflows from exchanges (positive net_flow) indicate accumulation.
    Inflows to exchanges (negative net_flow) indicate distribution.
    """

    def analyze(self, net_flow: float | None) -> dict[str, Any]:
        """Classify whale flow.

        Args:
            net_flow: Net USD flow (positive = outflows from exchanges = accumulation).
                None returns neutral.

        Returns:
            Dict with key ``whale_flow``.
        """
        if net_flow is None or abs(net_flow) < _THRESHOLD:
            return {"whale_flow": "neutral"}
        return {"whale_flow": "accumulating" if net_flow > 0 else "distributing"}
