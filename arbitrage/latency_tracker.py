"""
Latency Tracker — monitors API response times per exchange.
"""

import time
from collections import defaultdict, deque


class LatencyTracker:
    """Tracks API response latency for each exchange."""

    def __init__(self, max_samples: int = 100):
        self._samples: dict[str, deque] = defaultdict(lambda: deque(maxlen=max_samples))

    def record(self, exchange: str, latency_ms: float) -> None:
        """Record a latency measurement."""
        self._samples[exchange].append(latency_ms)

    def get_stats(self, exchange: str) -> dict:
        """Get latency statistics for an exchange."""
        samples = list(self._samples.get(exchange, []))
        if not samples:
            return {"exchange": exchange, "samples": 0}

        return {
            "exchange": exchange,
            "samples": len(samples),
            "avg_ms": round(sum(samples) / len(samples), 1),
            "min_ms": round(min(samples), 1),
            "max_ms": round(max(samples), 1),
            "last_ms": round(samples[-1], 1),
        }

    def get_all_stats(self) -> list[dict]:
        """Get latency stats for all tracked exchanges."""
        return [self.get_stats(ex) for ex in self._samples]
