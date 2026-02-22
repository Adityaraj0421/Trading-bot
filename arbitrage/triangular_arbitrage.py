"""
Triangular Arbitrage Engine v1.0
==================================
Detects and evaluates triangular arbitrage opportunities.

Triangular arb exploits price inconsistencies across 3 trading pairs
on the SAME exchange (no transfer delays).

Example: USDT → BTC → ETH → USDT
  If BTC/USDT = 100K, ETH/BTC = 0.035, ETH/USDT = 3600
  Forward: 1 USDT → 0.00001 BTC → 0.000286 ETH → 1.028 USDT (+2.8%)
  Reverse: 1 USDT → 0.000278 ETH → 0.00001 BTC → 1.002 USDT (+0.2%)

Also includes:
  - Cross-exchange latency monitoring
  - Execution time estimation
  - Multi-path triangular scanning
  - Fee-adjusted profitability
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class TriangularPath:
    """A triangular arbitrage path."""

    exchange: str
    path: list[str]  # e.g., ["BTC/USDT", "ETH/BTC", "ETH/USDT"]
    direction: str  # "forward" or "reverse"
    rates: list[float]  # Exchange rates along the path
    gross_return: float  # Return before fees
    fees_pct: float  # Total fees for 3 trades
    net_return: float  # Return after fees
    execution_time_ms: float  # Estimated execution time
    timestamp: float = 0.0

    @property
    def is_profitable(self) -> bool:
        """True if the path yields positive return after fees."""
        return self.net_return > 0

    def to_dict(self) -> dict:
        """Serialize triangular path to dict."""
        return {
            "exchange": self.exchange,
            "path": self.path,
            "direction": self.direction,
            "gross_return_pct": round(self.gross_return * 100, 4),
            "fees_pct": round(self.fees_pct * 100, 4),
            "net_return_pct": round(self.net_return * 100, 4),
            "execution_time_ms": round(self.execution_time_ms, 1),
            "profitable": self.is_profitable,
        }


@dataclass
class LatencyProfile:
    """Latency statistics for an exchange."""

    exchange: str
    avg_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    last_check: float = 0.0
    samples: int = 0


# ── Triangular Arbitrage Engine ───────────────────────────────────


class TriangularArbitrageEngine:
    """
    Scans for triangular arbitrage opportunities within a single exchange
    and estimates execution feasibility.
    """

    # Common triangular paths for crypto
    STANDARD_TRIANGLES = [
        ("BTC/USDT", "ETH/BTC", "ETH/USDT"),
        ("BTC/USDT", "SOL/BTC", "SOL/USDT"),
        ("BTC/USDT", "XRP/BTC", "XRP/USDT"),
        ("BTC/USDT", "BNB/BTC", "BNB/USDT"),
        ("ETH/USDT", "SOL/ETH", "SOL/USDT"),
        ("BTC/USDT", "DOGE/BTC", "DOGE/USDT"),
        ("BTC/USDT", "ADA/BTC", "ADA/USDT"),
        ("BTC/USDT", "LINK/BTC", "LINK/USDT"),
    ]

    # Minimum net profit to consider
    MIN_NET_PROFIT_PCT = 0.001  # 0.1%

    def __init__(self, exchange: Any = None, fee_pct: float = 0.001) -> None:
        """
        Args:
            exchange: CCXT exchange instance
            fee_pct: Per-trade fee (taker rate)
        """
        self.exchange = exchange
        self.fee_pct = fee_pct  # Fee per trade

        # Results
        self._opportunities: list[TriangularPath] = []
        self._scan_count: int = 0

        # Latency monitoring
        self._latencies: dict[str, deque] = {}  # exchange → deque of ms values
        self._latency_profiles: dict[str, LatencyProfile] = {}

    # ── Public Interface ──────────────────────────────────────────

    def scan_triangles(self, exchange_name: str = "binance") -> list[TriangularPath]:
        """
        Scan all standard triangular paths for opportunities.

        Returns list of profitable TriangularPath objects.
        """
        if self.exchange is None:
            return []

        self._scan_count += 1
        opportunities = []

        for triangle in self.STANDARD_TRIANGLES:
            try:
                # Fetch order books for all 3 pairs
                books = {}
                for pair in triangle:
                    t_start = time.time()
                    try:
                        book = self.exchange.fetch_order_book(pair, limit=5)
                        latency = (time.time() - t_start) * 1000
                        self._record_latency(exchange_name, latency)
                        books[pair] = book
                    except Exception as e:
                        __import__("logging").getLogger(__name__).debug("Order book fetch failed for %s: %s", pair, e)
                        break

                if len(books) != 3:
                    continue

                # Evaluate forward and reverse paths
                for direction in ["forward", "reverse"]:
                    path = self._evaluate_path(triangle, books, direction, exchange_name)
                    if path and path.is_profitable:
                        opportunities.append(path)

            except Exception as e:
                _log.debug("Triangle scan error for %s: %s", triangle, e)

        # Sort by net return
        opportunities.sort(key=lambda p: p.net_return, reverse=True)
        self._opportunities = opportunities
        return opportunities

    def scan_triangles_offline(self, prices: dict, exchange_name: str = "binance") -> list[TriangularPath]:
        """
        Scan using pre-fetched price data (for backtesting/testing).

        Args:
            prices: {pair: {"bid": float, "ask": float}}
        """
        self._scan_count += 1
        opportunities = []

        for triangle in self.STANDARD_TRIANGLES:
            if not all(p in prices for p in triangle):
                continue

            books = {}
            for pair in triangle:
                p = prices[pair]
                books[pair] = {
                    "bids": [[p["bid"], 1.0]],
                    "asks": [[p["ask"], 1.0]],
                }

            for direction in ["forward", "reverse"]:
                path = self._evaluate_path(triangle, books, direction, exchange_name)
                if path and path.is_profitable:
                    opportunities.append(path)

        opportunities.sort(key=lambda p: p.net_return, reverse=True)
        self._opportunities = opportunities
        return opportunities

    def get_latency_profile(self, exchange: str) -> LatencyProfile:
        """Get latency statistics for an exchange."""
        return self._latency_profiles.get(
            exchange,
            LatencyProfile(exchange=exchange),
        )

    def get_status(self) -> dict[str, Any]:
        """Dashboard status."""
        return {
            "scan_count": self._scan_count,
            "active_opportunities": len(self._opportunities),
            "opportunities": [o.to_dict() for o in self._opportunities[:5]],
            "latency_profiles": {
                k: {"avg_ms": round(v.avg_latency_ms, 1), "p95_ms": round(v.p95_latency_ms, 1), "samples": v.samples}
                for k, v in self._latency_profiles.items()
            },
        }

    # ── Internal: Path Evaluation ─────────────────────────────────

    def _evaluate_path(self, triangle: tuple, books: dict, direction: str, exchange: str) -> TriangularPath | None:
        """
        Evaluate a triangular path for profitability.

        Forward: USDT → A → B → USDT
        Reverse: USDT → B → A → USDT
        """
        pair1, pair2, pair3 = triangle

        try:
            if direction == "forward":
                # Step 1: Buy pair1 (e.g., buy BTC with USDT at ask)
                r1 = 1.0 / books[pair1]["asks"][0][0]  # USDT → BTC
                # Step 2: Buy pair2 using BTC (e.g., BTC → ETH at ask)
                # pair2 is ETH/BTC, so selling BTC = buying ETH
                r2 = 1.0 / books[pair2]["asks"][0][0]  # BTC → ETH
                # Step 3: Sell pair3 (e.g., sell ETH for USDT at bid)
                r3 = books[pair3]["bids"][0][0]  # ETH → USDT

                # Chain: start_USDT × r1 × r2 × r3 = end_USDT
                gross = r1 * r2 * r3
                # But we need to normalize: if starting with 1 USDT
                # r1 gives us BTC amount, r2 converts that to ETH amount
                # Actually for forward path:
                # 1 USDT → (1/ask1) BTC → (1/ask1)(1/ask2) ETH → (1/ask1)(1/ask2)(bid3) USDT
                rates = [r1, r2, r3]

            else:
                # Reverse: USDT → ETH → BTC → USDT
                # Step 1: Buy pair3 (buy ETH with USDT at ask)
                r1 = 1.0 / books[pair3]["asks"][0][0]  # USDT → ETH
                # Step 2: Sell pair2 (sell ETH for BTC at bid)
                r2 = books[pair2]["bids"][0][0]  # ETH → BTC
                # Step 3: Sell pair1 (sell BTC for USDT at bid)
                r3 = books[pair1]["bids"][0][0]  # BTC → USDT

                gross = r1 * r2 * r3
                rates = [r1, r2, r3]

            # Compute gross return
            gross_return = gross - 1.0

            # Fees: 3 trades × fee_pct each
            total_fees = 3 * self.fee_pct
            # Compounding: (1 - fee)^3
            fee_adjusted = gross * (1 - self.fee_pct) ** 3
            net_return = fee_adjusted - 1.0

            # Estimate execution time (3 trades)
            latency = self.get_latency_profile(exchange)
            exec_time = latency.avg_latency_ms * 3 if latency.avg_latency_ms > 0 else 300

            if net_return < self.MIN_NET_PROFIT_PCT:
                return None

            return TriangularPath(
                exchange=exchange,
                path=list(triangle),
                direction=direction,
                rates=[round(r, 8) for r in rates],
                gross_return=gross_return,
                fees_pct=total_fees,
                net_return=net_return,
                execution_time_ms=exec_time,
                timestamp=time.time(),
            )

        except (IndexError, ZeroDivisionError, KeyError):
            return None

    # ── Internal: Latency Monitoring ──────────────────────────────

    def _record_latency(self, exchange: str, latency_ms: float) -> None:
        """Record a latency sample for an exchange."""
        if exchange not in self._latencies:
            self._latencies[exchange] = deque(maxlen=100)

        self._latencies[exchange].append(latency_ms)

        # Update profile
        samples = list(self._latencies[exchange])
        n = len(samples)
        sorted_samples = sorted(samples)

        self._latency_profiles[exchange] = LatencyProfile(
            exchange=exchange,
            avg_latency_ms=sum(samples) / n,
            p95_latency_ms=sorted_samples[int(n * 0.95)] if n >= 20 else sorted_samples[-1],
            p99_latency_ms=sorted_samples[int(n * 0.99)] if n >= 100 else sorted_samples[-1],
            last_check=time.time(),
            samples=n,
        )
