"""
Funding Rate Arbitrage Engine v1.0
====================================
Delta-neutral funding rate harvesting strategy.
Collects funding payments by being simultaneously long spot + short perpetual
(or vice versa) when funding rates are persistently elevated.

Strategy:
  - When funding > +0.03%: Long spot + Short perp (collect from longs)
  - When funding < -0.03%: Short spot + Long perp (collect from shorts)
  - Unwind when funding normalizes to near-zero
  - Auto-sizes based on funding magnitude and historical stability

This is a market-neutral strategy — profit comes from funding payments,
not price movement.
"""

import time
import logging
import requests
import numpy as np
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque
from config import Config

_log = logging.getLogger(__name__)


@dataclass
class FundingPosition:
    """Represents an active delta-neutral funding arbitrage position."""
    pair: str
    direction: str       # "long_basis" (long spot, short perp) or "short_basis"
    spot_entry: float    # Spot entry price
    perp_entry: float    # Perp entry price
    size_usd: float      # Position size in USD
    entry_funding: float # Funding rate at entry
    entry_time: datetime
    total_funding_collected: float = 0.0
    funding_payments: int = 0
    last_funding_time: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "pair": self.pair,
            "direction": self.direction,
            "spot_entry": self.spot_entry,
            "perp_entry": self.perp_entry,
            "size_usd": round(self.size_usd, 2),
            "entry_funding_rate": self.entry_funding,
            "total_funding_collected": round(self.total_funding_collected, 4),
            "funding_payments": self.funding_payments,
            "entry_time": self.entry_time.isoformat(),
            "annualized_yield_pct": round(self.annualized_yield_pct, 2),
        }

    @property
    def annualized_yield_pct(self) -> float:
        if self.size_usd <= 0:
            return 0.0
        hours_held = max(1, (datetime.now() - self.entry_time).total_seconds() / 3600)
        return_pct = (self.total_funding_collected / self.size_usd)
        return return_pct * (8760 / hours_held) * 100


@dataclass
class FundingSnapshot:
    """Point-in-time funding rate data."""
    pair: str
    funding_rate: float      # Current 8h funding rate
    predicted_rate: float    # Predicted next funding rate
    next_funding_time: str
    timestamp: float


class FundingArbitrageEngine:
    """
    Delta-neutral funding rate arbitrage engine.
    Monitors perpetual funding rates and opens/manages hedged positions
    to harvest funding payments as yield.
    """

    # --- Entry/exit thresholds ---
    MIN_FUNDING_RATE = 0.0003    # 0.03% per 8h — minimum to enter a position
    EXIT_FUNDING_RATE = 0.0001   # 0.01% per 8h — close when funding drops here
    MAX_POSITION_SIZE_PCT = 0.15 # 15% of capital per funding arb pair
    STABILITY_LOOKBACK = 10      # Minimum consistent funding readings before entry
    MIN_STABILITY = 0.5          # Minimum stability score (0-1) to enter
    MIN_PAYMENTS_BEFORE_EXIT = 3 # Collect at least N payments before allowing negative-yield exit

    # Funding rate math constants
    FUNDING_PERIODS_PER_DAY = 3  # Binance: 3 × 8h funding periods
    ROUND_TRIP_FEE_MULTIPLIER = 4  # 2 entries + 2 exits

    # History tracking
    FUNDING_HISTORY_MAXLEN = 100

    BINANCE_FUTURES_API = "https://fapi.binance.com"

    def __init__(self, symbols: list[str] = None, capital: float = None):
        self.symbols = symbols or [
            p.replace("/", "") for p in Config.TRADING_PAIRS
        ]
        self.capital = capital or Config.INITIAL_CAPITAL
        self.positions: list[FundingPosition] = []
        self._funding_history: dict[str, deque] = {
            s: deque(maxlen=self.FUNDING_HISTORY_MAXLEN) for s in self.symbols
        }
        self._last_scan: dict = {}
        self._total_yield: float = 0.0

    def fetch_funding_rates(self) -> dict[str, FundingSnapshot]:
        """Fetch current funding rates from Binance futures."""
        rates = {}
        for symbol in self.symbols:
            try:
                # Current funding rate
                resp = requests.get(
                    f"{self.BINANCE_FUTURES_API}/fapi/v1/premiumIndex",
                    params={"symbol": symbol},
                    timeout=5,
                )
                if resp.status_code == 200:
                    data = resp.json()
                    snapshot = FundingSnapshot(
                        pair=symbol,
                        funding_rate=float(data.get("lastFundingRate", 0)),
                        predicted_rate=float(data.get("nextFundingRate", 0) or 0),
                        next_funding_time=str(data.get("nextFundingTime", "")),
                        timestamp=time.time(),
                    )
                    rates[symbol] = snapshot

                    # Track history
                    if symbol not in self._funding_history:
                        self._funding_history[symbol] = deque(maxlen=self.FUNDING_HISTORY_MAXLEN)
                    self._funding_history[symbol].append(snapshot)

            except Exception as e:
                _log.warning(f"Funding rate fetch error for {symbol}: {e}")

        self._last_scan = {k: v.funding_rate for k, v in rates.items()}
        return rates

    def analyze_opportunities(self) -> list[dict]:
        """
        Analyze funding rates to find arbitrage opportunities.
        Returns list of opportunities ranked by annualized yield.
        """
        rates = self.fetch_funding_rates()
        opportunities = []

        for symbol, snapshot in rates.items():
            fr = snapshot.funding_rate
            abs_fr = abs(fr)

            if abs_fr < self.MIN_FUNDING_RATE:
                continue

            # Check funding stability
            history = list(self._funding_history.get(symbol, []))
            stability = self._compute_stability(history)

            if stability < self.MIN_STABILITY:
                continue  # Too volatile, funding might flip

            # Compute annualized yield
            annualized_yield = abs_fr * self.FUNDING_PERIODS_PER_DAY * 365 * 100  # As percentage

            # Deduct estimated costs (entry + exit fees)
            cost_pct = Config.FEE_PCT * self.ROUND_TRIP_FEE_MULTIPLIER
            net_annualized = annualized_yield - (cost_pct * 365 * 100)

            if net_annualized <= 0:
                continue

            direction = "long_basis" if fr > 0 else "short_basis"

            # Check if we already have a position
            has_position = any(p.pair == symbol for p in self.positions)

            opportunities.append({
                "pair": symbol,
                "funding_rate": fr,
                "predicted_rate": snapshot.predicted_rate,
                "direction": direction,
                "annualized_yield_pct": round(annualized_yield, 2),
                "net_annualized_yield_pct": round(net_annualized, 2),
                "stability": round(stability, 2),
                "has_position": has_position,
                "recommended_size_usd": round(
                    self.capital * self.MAX_POSITION_SIZE_PCT * stability, 2
                ),
            })

        # Sort by net yield
        opportunities.sort(
            key=lambda x: x["net_annualized_yield_pct"], reverse=True
        )
        return opportunities

    def _compute_stability(self, history: list[FundingSnapshot]) -> float:
        """
        Compute funding rate stability (0-1).
        High stability = funding consistently in same direction.
        """
        if len(history) < 3:
            return 0.0

        rates = [s.funding_rate for s in history]

        # Direction consistency: what % of readings are same sign as latest?
        latest_sign = 1 if rates[-1] > 0 else -1
        same_sign = sum(1 for r in rates if (r > 0) == (latest_sign > 0))
        direction_consistency = same_sign / len(rates)

        # Magnitude consistency: coefficient of variation (lower = more stable)
        abs_rates = [abs(r) for r in rates if r != 0]
        if abs_rates:
            cv = np.std(abs_rates) / max(np.mean(abs_rates), 1e-10)
            magnitude_stability = max(0, 1 - cv)
        else:
            magnitude_stability = 0.0

        return (direction_consistency * 0.7 + magnitude_stability * 0.3)

    def open_position(self, pair: str, spot_price: float,
                      perp_price: float, funding_rate: float,
                      size_usd: float = None) -> FundingPosition:
        """Open a delta-neutral funding arbitrage position."""
        size = size_usd or (self.capital * self.MAX_POSITION_SIZE_PCT * 0.5)

        direction = "long_basis" if funding_rate > 0 else "short_basis"

        pos = FundingPosition(
            pair=pair,
            direction=direction,
            spot_entry=spot_price,
            perp_entry=perp_price,
            size_usd=size,
            entry_funding=funding_rate,
            entry_time=datetime.now(),
        )
        self.positions.append(pos)

        _log.info(
            f"[FundingArb] OPENED {direction} {pair}: "
            f"size=${size:,.2f}, funding={funding_rate:.4%}"
        )
        return pos

    def process_funding_payment(self, pair: str,
                                funding_rate: float) -> float:
        """Record a funding payment for active position."""
        for pos in self.positions:
            if pos.pair == pair:
                # We collect funding when our direction matches
                if (pos.direction == "long_basis" and funding_rate > 0) or \
                   (pos.direction == "short_basis" and funding_rate < 0):
                    payment = abs(funding_rate) * pos.size_usd
                else:
                    payment = -abs(funding_rate) * pos.size_usd

                pos.total_funding_collected += payment
                pos.funding_payments += 1
                pos.last_funding_time = datetime.now()
                self._total_yield += payment
                return payment
        return 0.0

    def check_exits(self) -> list[dict]:
        """Check if any positions should be unwound."""
        exits = []
        rates = self._last_scan

        for pos in list(self.positions):
            current_rate = rates.get(pos.pair, 0)
            abs_rate = abs(current_rate)

            should_exit = False
            reason = ""

            # Exit if funding dropped below threshold
            if abs_rate < self.EXIT_FUNDING_RATE:
                should_exit = True
                reason = "funding_normalized"

            # Exit if funding flipped direction
            if pos.direction == "long_basis" and current_rate < 0:
                should_exit = True
                reason = "funding_flipped"
            elif pos.direction == "short_basis" and current_rate > 0:
                should_exit = True
                reason = "funding_flipped"

            # Exit if net yield turned negative
            if pos.total_funding_collected < 0 and pos.funding_payments >= self.MIN_PAYMENTS_BEFORE_EXIT:
                should_exit = True
                reason = "negative_yield"

            if should_exit:
                exits.append({
                    "pair": pos.pair,
                    "reason": reason,
                    "total_collected": round(pos.total_funding_collected, 4),
                    "payments": pos.funding_payments,
                    "annualized_yield": round(pos.annualized_yield_pct, 2),
                })
                self.positions.remove(pos)
                _log.info(
                    f"[FundingArb] CLOSED {pos.pair}: "
                    f"reason={reason}, collected=${pos.total_funding_collected:.4f}"
                )

        return exits

    def get_status(self) -> dict:
        """Get funding arbitrage engine status."""
        return {
            "active_positions": len(self.positions),
            "total_yield": round(self._total_yield, 4),
            "positions": [p.to_dict() for p in self.positions],
            "current_rates": {
                k: f"{v:.4%}" for k, v in self._last_scan.items()
            },
            "symbols_monitored": self.symbols,
        }

    def to_dict(self) -> dict:
        return {
            "positions": [p.to_dict() for p in self.positions],
            "total_yield": self._total_yield,
        }
