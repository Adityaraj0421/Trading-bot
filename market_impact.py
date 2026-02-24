"""
Realistic Market Impact Model v1.0
====================================
Replaces fixed slippage with dynamic, realistic cost modeling.

Components:
  1. Volatility-correlated slippage: slippage = base + (atr_pct × mult)
  2. Volume-dependent market impact: impact = sqrt(order_size / avg_volume) × const
  3. Partial fill simulation: large orders may not fill entirely
  4. Spread modeling: wider spreads during low-liquidity periods
  5. Stress scenario injection: flash crash, exchange lag, liquidity drought

Research basis:
  - Almgren-Chriss market impact model
  - Kyle's lambda (price impact coefficient)
  - Empirical crypto market impact studies (2024-2025)
"""

from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Any

import numpy as np

_log = logging.getLogger(__name__)


# ── Data Classes ──────────────────────────────────────────────────


@dataclass
class ExecutionResult:
    """Result of simulated order execution."""

    requested_quantity: float
    filled_quantity: float  # May be less than requested
    fill_rate: float  # filled / requested (0-1)
    average_fill_price: float  # Actual average execution price
    slippage_pct: float  # Total slippage %
    spread_cost_pct: float  # Cost from bid-ask spread
    market_impact_pct: float  # Price impact from order size
    total_cost_pct: float  # slippage + spread + impact + fees
    fees_paid: float
    is_partial_fill: bool

    def to_dict(self) -> dict[str, Any]:
        """Serialize execution result to a rounded dictionary for logging."""
        return {
            "filled_qty": round(self.filled_quantity, 6),
            "fill_rate": round(self.fill_rate, 4),
            "avg_fill_price": round(self.average_fill_price, 2),
            "slippage_pct": round(self.slippage_pct * 100, 4),
            "spread_cost_pct": round(self.spread_cost_pct * 100, 4),
            "impact_pct": round(self.market_impact_pct * 100, 4),
            "total_cost_pct": round(self.total_cost_pct * 100, 4),
            "partial_fill": self.is_partial_fill,
        }


@dataclass
class StressScenario:
    """A market stress scenario for backtesting."""

    name: str
    price_shock_pct: float  # Sudden price move
    liquidity_mult: float  # 0.1 = 90% liquidity drop
    spread_mult: float  # How much spread widens
    latency_ms: float  # Added execution delay
    duration_bars: int  # How long it lasts
    probability: float  # Probability per bar (0-1)


# ── Market Impact Model ──────────────────────────────────────────


class MarketImpactModel:
    """
    Dynamic, realistic execution cost model.

    Replaces fixed slippage with:
    - ATR-based volatility slippage
    - Square-root market impact (Almgren-Chriss)
    - Volume-dependent fill simulation
    - Time-of-day liquidity adjustment
    """

    # Base parameters (calibrated for BTC/USDT on Binance)
    BASE_SLIPPAGE_PCT = 0.0002  # 0.02% base slippage
    VOLATILITY_SLIPPAGE_MULT = 0.5  # ATR multiplier for vol-adjusted slippage
    IMPACT_COEFFICIENT = 0.1  # Kyle's lambda (price impact per sqrt-volume)
    BASE_SPREAD_PCT = 0.0001  # 0.01% base spread (tight for BTC)
    PARTIAL_FILL_THRESHOLD = 0.05  # Orders > 5% of avg volume may partially fill

    # Stress scenarios
    STRESS_SCENARIOS = [
        StressScenario("flash_crash", -0.15, 0.1, 10.0, 5000, 3, 0.001),
        StressScenario("liquidity_drought", 0.0, 0.2, 5.0, 2000, 10, 0.005),
        StressScenario("exchange_lag", 0.0, 0.8, 2.0, 10000, 5, 0.003),
        StressScenario("fat_finger", 0.05, 0.5, 3.0, 1000, 1, 0.0005),
        StressScenario("whale_dump", -0.08, 0.3, 8.0, 3000, 5, 0.002),
    ]

    def __init__(self, fee_pct: float = 0.001, enable_partial_fills: bool = True, enable_stress: bool = True) -> None:
        """Initialise the market impact model.

        Args:
            fee_pct: Exchange fee rate applied to each execution (e.g.
                ``0.001`` for 0.1 %).
            enable_partial_fills: When ``True``, large orders relative to
                bar volume may receive fractional fills.
            enable_stress: When ``True``, random stress scenarios are
                injected each bar according to their configured probability.
        """
        self.fee_pct = fee_pct
        self.enable_partial_fills = enable_partial_fills
        self.enable_stress = enable_stress

        # State
        self._active_stress: StressScenario | None = None
        self._stress_bars_remaining: int = 0
        self._execution_log: deque = deque(maxlen=500)

    # ── Public Interface ──────────────────────────────────────────

    def simulate_execution(
        self,
        price: float,
        quantity: float,
        side: str,
        is_entry: bool,
        atr_pct: float = 0.01,
        avg_volume: float = 1000.0,
        bar_volume: float = 100.0,
    ) -> ExecutionResult:
        """
        Simulate realistic order execution.

        Args:
            price: Current market price.
            quantity: Desired order size in base asset units.
            side: Trade direction — ``"long"`` or ``"short"``.
            is_entry: ``True`` for position entries, ``False`` for exits.
                Entries receive adverse fill prices; exits receive
                more-favourable fills.
            atr_pct: ATR expressed as a fraction of price (volatility
                proxy). Drives the volatility-slippage component.
            avg_volume: Average bar volume in base asset units. Used for
                market-impact and partial-fill calculations.
            bar_volume: Current bar volume. Used for spread and fill-rate
                calculations.

        Returns:
            ``ExecutionResult`` dataclass containing filled quantity, fill
            rate, average fill price, individual cost components
            (slippage, spread, market impact), total cost as a fraction of
            price, fees paid in quote currency, and a partial-fill flag.
        """
        # Check for active stress scenario
        stress_mult = self._get_stress_multipliers()

        # 1. Spread cost
        spread_pct = self._compute_spread(atr_pct, bar_volume, avg_volume, stress_mult)

        # 2. Volatility-correlated slippage
        slippage_pct = self._compute_slippage(atr_pct, stress_mult)

        # 3. Market impact (square-root model)
        impact_pct = self._compute_market_impact(quantity, avg_volume, stress_mult)

        # 4. Partial fill simulation
        fill_rate = self._simulate_fill_rate(quantity, bar_volume, avg_volume, stress_mult)
        filled_qty = quantity * fill_rate

        # 5. Compute actual fill price
        # Direction: entries get worse prices, exits get better
        total_adverse_pct = spread_pct / 2 + slippage_pct + impact_pct
        if is_entry:
            if side == "long":
                fill_price = price * (1 + total_adverse_pct)
            else:
                fill_price = price * (1 - total_adverse_pct)
        else:
            if side == "long":
                fill_price = price * (1 - total_adverse_pct)
            else:
                fill_price = price * (1 + total_adverse_pct)

        # 6. Fees
        fees = fill_price * filled_qty * self.fee_pct

        # Total cost
        total_cost_pct = spread_pct / 2 + slippage_pct + impact_pct + self.fee_pct

        result = ExecutionResult(
            requested_quantity=quantity,
            filled_quantity=filled_qty,
            fill_rate=fill_rate,
            average_fill_price=fill_price,
            slippage_pct=slippage_pct,
            spread_cost_pct=spread_pct,
            market_impact_pct=impact_pct,
            total_cost_pct=total_cost_pct,
            fees_paid=fees,
            is_partial_fill=fill_rate < 0.999,
        )

        self._execution_log.append(result.to_dict())
        return result

    def advance_bar(self) -> None:
        """Advance the internal bar counter and manage stress scenario state.

        Should be called once per completed bar. Decrements the active
        stress duration counter and, when the stress period ends, clears the
        active scenario. May also probabilistically activate a new stress
        scenario from ``STRESS_SCENARIOS`` when ``enable_stress`` is
        ``True``.
        """
        if self._stress_bars_remaining > 0:
            self._stress_bars_remaining -= 1
            if self._stress_bars_remaining <= 0:
                self._active_stress = None

        # Random stress injection
        if self.enable_stress and self._active_stress is None:
            for scenario in self.STRESS_SCENARIOS:
                if np.random.random() < scenario.probability:
                    self._active_stress = scenario
                    self._stress_bars_remaining = scenario.duration_bars
                    _log.debug("Stress scenario activated: %s", scenario.name)
                    break

    def get_active_stress(self) -> dict[str, Any] | None:
        """Return metadata about the currently active stress scenario.

        Returns:
            Dictionary with keys ``name``, ``bars_remaining``,
            ``liquidity_mult``, and ``spread_mult`` when a stress scenario
            is active, or ``None`` when the market is in a normal state.
        """
        if self._active_stress is None:
            return None
        return {
            "name": self._active_stress.name,
            "bars_remaining": self._stress_bars_remaining,
            "liquidity_mult": self._active_stress.liquidity_mult,
            "spread_mult": self._active_stress.spread_mult,
        }

    def get_execution_stats(self) -> dict[str, Any]:
        """Aggregate execution statistics from the rolling log.

        Returns:
            Dictionary with keys:

            - ``executions`` (int): Total number of simulated executions.
            - ``avg_slippage_pct`` (float): Mean slippage across all fills.
            - ``avg_impact_pct`` (float): Mean market-impact cost.
            - ``avg_fill_rate`` (float): Mean fill rate (0–1).
            - ``partial_fills`` (int): Count of partial fills.
            - ``partial_fill_rate`` (float): Fraction of fills that were
              partial.
            - ``max_slippage_pct`` (float): Worst-case slippage observed.

            Returns ``{"executions": 0}`` when the log is empty.
        """
        log = list(self._execution_log)
        if not log:
            return {"executions": 0}

        slippages = [e["slippage_pct"] for e in log]
        impacts = [e["impact_pct"] for e in log]
        fills = [e["fill_rate"] for e in log]
        partials = sum(1 for e in log if e["partial_fill"])

        return {
            "executions": len(log),
            "avg_slippage_pct": round(np.mean(slippages), 4),
            "avg_impact_pct": round(np.mean(impacts), 4),
            "avg_fill_rate": round(np.mean(fills), 4),
            "partial_fills": partials,
            "partial_fill_rate": round(partials / len(log), 4),
            "max_slippage_pct": round(max(slippages), 4),
        }

    # ── Internal: Cost Components ─────────────────────────────────

    def _compute_slippage(self, atr_pct: float, stress_mult: dict[str, float]) -> float:
        """Compute volatility-correlated slippage.

        ``slippage = base_slippage × spread_mult + atr_pct × VOLATILITY_SLIPPAGE_MULT``

        Higher volatility means prices move more during order execution.

        Args:
            atr_pct: ATR as a fraction of the current price.
            stress_mult: Active stress multipliers; the ``"spread"`` key
                scales the base slippage component.

        Returns:
            Slippage as a fraction of price (e.g. ``0.001`` = 0.1 %).
        """
        base = self.BASE_SLIPPAGE_PCT * stress_mult.get("spread", 1.0)
        vol_component = atr_pct * self.VOLATILITY_SLIPPAGE_MULT
        return base + vol_component

    def _compute_market_impact(self, quantity: float, avg_volume: float, stress_mult: dict[str, float]) -> float:
        """Compute square-root market impact (Almgren-Chriss model).

        ``impact = IMPACT_COEFFICIENT × sqrt(quantity / effective_volume)``

        Larger orders relative to available volume have greater price impact.
        Effective volume is reduced by the liquidity stress multiplier.

        Args:
            quantity: Order size in base asset units.
            avg_volume: Average bar volume in base asset units.
            stress_mult: Active stress multipliers; the ``"liquidity"`` key
                scales available volume down during stress events.

        Returns:
            Market impact as a fraction of price, capped at 5 %.
        """
        if avg_volume <= 0:
            return 0.001

        # Adjust volume for liquidity stress
        effective_volume = avg_volume * stress_mult.get("liquidity", 1.0)
        participation_rate = quantity / (effective_volume + 1e-10)

        impact = self.IMPACT_COEFFICIENT * np.sqrt(participation_rate)
        return min(impact, 0.05)  # Cap at 5%

    def _compute_spread(
        self, atr_pct: float, bar_volume: float, avg_volume: float, stress_mult: dict[str, float]
    ) -> float:
        """Compute the dynamic bid-ask spread cost.

        Spread widens during low-liquidity bars, high-volatility periods,
        and active stress scenarios.

        Args:
            atr_pct: ATR as a fraction of price; spreads above 3 % ATR
                are widened further.
            bar_volume: Current bar volume in base asset units.
            avg_volume: Average bar volume. Used to compute a
                volume-ratio that adjusts the spread.
            stress_mult: Active stress multipliers; the ``"spread"`` key
                is applied as a final multiplier.

        Returns:
            Full bid-ask spread as a fraction of price, capped at 2 %.
        """
        base = self.BASE_SPREAD_PCT

        # Volume-adjusted: low volume = wider spread
        if avg_volume > 0 and bar_volume > 0:
            vol_ratio = bar_volume / avg_volume
            if vol_ratio < 0.5:
                base *= 2.0  # Double spread in low-volume bars
            elif vol_ratio < 0.2:
                base *= 5.0  # 5x spread in very low volume
        else:
            base *= 1.5

        # Volatility adjustment
        if atr_pct > 0.03:
            base *= 1 + (atr_pct - 0.03) * 10

        # Stress multiplier
        base *= stress_mult.get("spread", 1.0)

        return min(base, 0.02)  # Cap at 2%

    def _simulate_fill_rate(
        self, quantity: float, bar_volume: float, avg_volume: float, stress_mult: dict[str, float]
    ) -> float:
        """Simulate partial fills for large orders.

        Orders whose size is a significant fraction of bar volume may not
        fill entirely. Fill rate degrades linearly beyond
        ``PARTIAL_FILL_THRESHOLD`` and asymptotes at 50 % for massive
        orders.

        Args:
            quantity: Order size in base asset units.
            bar_volume: Current bar volume in base asset units.
            avg_volume: Average bar volume (unused directly; see
                ``enable_partial_fills`` guard).
            stress_mult: Active stress multipliers; the ``"liquidity"`` key
                scales the effective bar volume.

        Returns:
            Fill rate as a fraction in ``(0, 1]``. Returns ``1.0`` when
            ``enable_partial_fills`` is ``False`` or when the order is
            small relative to bar volume.
        """
        if not self.enable_partial_fills:
            return 1.0

        if avg_volume <= 0 or bar_volume <= 0:
            return 0.9

        effective_volume = bar_volume * stress_mult.get("liquidity", 1.0)
        participation = quantity / (effective_volume + 1e-10)

        if participation < self.PARTIAL_FILL_THRESHOLD:
            return 1.0  # Small order, full fill
        elif participation < 0.2:
            # Partial fill: linear degradation
            return 1.0 - (participation - self.PARTIAL_FILL_THRESHOLD) * 2
        elif participation < 0.5:
            return 0.7  # Large order, poor fill
        else:
            return 0.5  # Massive order, only half fills

    def _get_stress_multipliers(self) -> dict[str, float]:
        """Return the spread and liquidity multipliers for the active stress scenario.

        Returns:
            Dictionary with ``"spread"`` and ``"liquidity"`` keys. Both
            default to ``1.0`` when no stress scenario is active.
        """
        if self._active_stress is None:
            return {"spread": 1.0, "liquidity": 1.0}

        return {
            "spread": self._active_stress.spread_mult,
            "liquidity": self._active_stress.liquidity_mult,
        }
