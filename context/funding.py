"""FundingAnalyzer — classify perpetual funding rate as market pressure signal."""

from __future__ import annotations

from typing import Any

# Thresholds (8h funding rate as decimal)
_MILD_THRESHOLD = 0.0003      # 0.03% per 8h = long crowded (mild)
_EXTREME_THRESHOLD = 0.0010   # 0.10% per 8h = long crowded (extreme)
_SHORT_MILD = -0.0002         # -0.02% per 8h = short crowded (mild)
_SHORT_EXTREME = -0.0005      # -0.05% per 8h = short crowded (extreme)


class FundingAnalyzer:
    """Classifies current funding rate into a FundingPressure Literal.

    Positive funding → longs pay shorts → long-crowded.
    Negative funding → shorts pay longs → short-crowded.
    """

    def analyze(self, funding_rate: float | None) -> dict[str, Any]:
        """Classify funding rate.

        Args:
            funding_rate: Current 8h funding rate as a decimal (e.g. 0.0005 = 0.05%).
                None returns neutral.

        Returns:
            Dict with key ``funding_pressure``.
        """
        if funding_rate is None:
            return {"funding_pressure": "neutral"}

        if funding_rate >= _EXTREME_THRESHOLD:
            return {"funding_pressure": "long_crowded_extreme"}
        if funding_rate >= _MILD_THRESHOLD:
            return {"funding_pressure": "long_crowded_mild"}
        if funding_rate <= _SHORT_EXTREME:
            return {"funding_pressure": "short_crowded_extreme"}
        if funding_rate <= _SHORT_MILD:
            return {"funding_pressure": "short_crowded_mild"}
        return {"funding_pressure": "neutral"}
