"""
Risk Management module v4.1
============================
v4.1: Thread-safe position management with atomic check-and-open.
v4.0: Volatility-adaptive position sizing, tiered drawdown protocol,
      correlation-adjusted sizing, volatility targeting (15% annual).
v3.0: Trailing stop-loss, transaction cost awareness, max hold duration,
      position timeout, enhanced portfolio tracking.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any

import numpy as np

from config import Config

_log = logging.getLogger(__name__)


@dataclass
class Position:
    """An open trading position with trailing-stop tracking."""

    symbol: str
    side: str  # "long" or "short"
    entry_price: float
    quantity: float
    entry_time: datetime
    stop_loss: float
    take_profit: float
    strategy_name: str = ""
    # Trailing stop fields
    trailing_stop: float = 0.0
    highest_price: float = 0.0  # Track high water mark (longs)
    lowest_price: float = 0.0  # Track low water mark (shorts)
    entry_bar: int = 0  # Bar count at entry (for max hold)
    # v9.1: Breakeven stop — set True once SL has been moved to entry
    breakeven_triggered: bool = False

    def __post_init__(self) -> None:
        """Finalize position initialization after the dataclass ``__init__``.

        Sets ``highest_price`` and ``lowest_price`` to ``entry_price`` when
        they are still at their default of ``0.0``, and initializes
        ``trailing_stop`` to the entry-price-based value derived from
        ``Config.TRAILING_STOP_PCT`` when it has not been explicitly provided.
        """
        if self.highest_price == 0.0:
            self.highest_price = self.entry_price
        if self.lowest_price == 0.0:
            self.lowest_price = self.entry_price
        if self.trailing_stop == 0.0:
            trail_pct = Config.TRAILING_STOP_PCT
            if self.side == "long":
                self.trailing_stop = self.entry_price * (1 - trail_pct)
            else:
                self.trailing_stop = self.entry_price * (1 + trail_pct)

    @property
    def notional_value(self) -> float:
        """Return the notional value (entry_price * quantity) of the position."""
        return self.entry_price * self.quantity

    def update_trailing_stop(
        self,
        current_high: float,
        current_low: float,
        trail_pct_override: float | None = None,
        atr_pct: float | None = None,
    ) -> None:
        """Update the trailing stop based on new intra-bar price extremes.

        When ``atr_pct`` is provided the trailing distance equals
        ``Config.ATR_TRAILING_MULT × atr_pct`` (adaptive to current
        volatility). Otherwise falls back to ``trail_pct_override`` or
        ``Config.TRAILING_STOP_PCT``.

        Trailing stop only moves in the favorable direction (ratchet effect):
        upward for longs, downward for shorts.

        Args:
            current_high: Highest price seen in the current bar.
            current_low: Lowest price seen in the current bar.
            trail_pct_override: Fixed trailing percentage to use when
                ``atr_pct`` is not provided (e.g. ``0.015`` for 1.5%).
            atr_pct: ATR expressed as a fraction of the current price.
                When non-zero, takes precedence over ``trail_pct_override``.
        """
        if atr_pct and atr_pct > 0:
            trail_pct = Config.ATR_TRAILING_MULT * atr_pct
        else:
            trail_pct = trail_pct_override or Config.TRAILING_STOP_PCT
        if self.side == "long":
            if current_high > self.highest_price:
                self.highest_price = current_high
                new_trail = self.highest_price * (1 - trail_pct)
                self.trailing_stop = max(self.trailing_stop, new_trail)
        else:
            if current_low < self.lowest_price:
                self.lowest_price = current_low
                new_trail = self.lowest_price * (1 + trail_pct)
                self.trailing_stop = min(self.trailing_stop, new_trail)

    def check_breakeven(self, current_price: float, fee_pct: float) -> bool:
        """
        Move stop-loss to breakeven (entry + fee buffer) once unrealized PnL
        exceeds BREAKEVEN_TRIGGER_PCT of the take-profit distance.

        Only triggers once per position (breakeven_triggered flag prevents re-entry).

        Args:
            current_price: Latest market price.
            fee_pct: Round-trip fee used as buffer above/below entry.

        Returns:
            True if the stop was newly moved to breakeven, False otherwise.
        """
        if self.breakeven_triggered:
            return False

        tp_distance = abs(self.take_profit - self.entry_price)
        if tp_distance <= 0:
            return False

        trigger = Config.BREAKEVEN_TRIGGER_PCT * tp_distance
        unrealized = self.unrealized_pnl(current_price)

        if unrealized >= trigger:
            buffer = self.entry_price * fee_pct
            if self.side == "long":
                new_sl = self.entry_price + buffer
                if new_sl > self.stop_loss:
                    self.stop_loss = new_sl
                    self.breakeven_triggered = True
                    _log.info(
                        "[Risk] Breakeven stop triggered for %s %s: SL → %.4f",
                        self.side.upper(),
                        self.symbol,
                        new_sl,
                    )
                    return True
            else:
                new_sl = self.entry_price - buffer
                if new_sl < self.stop_loss:
                    self.stop_loss = new_sl
                    self.breakeven_triggered = True
                    _log.info(
                        "[Risk] Breakeven stop triggered for %s %s: SL → %.4f",
                        self.side.upper(),
                        self.symbol,
                        new_sl,
                    )
                    return True
        return False

    def check_exit(self, current_price: float, current_bar: int = 0) -> str | None:
        """Check all exit conditions and return the first triggered reason.

        Evaluates fixed stop-loss, take-profit, trailing stop, and maximum
        hold duration in order. The trailing stop only fires when it is
        strictly more favorable than the fixed stop-loss.

        Args:
            current_price: Latest market price.
            current_bar: Current agent bar counter used to evaluate
                ``MAX_HOLD_BARS``. Pass ``0`` to skip max-duration check.

        Returns:
            Exit reason string (``"stop_loss"``, ``"take_profit"``,
            ``"trailing_stop"``, or ``"max_duration"``), or ``None`` if no
            exit condition is met.
        """
        if self.side == "long":
            if current_price <= self.stop_loss:
                return "stop_loss"
            if current_price >= self.take_profit:
                return "take_profit"
            # Trailing stop only triggers if above the fixed stop
            if current_price <= self.trailing_stop and self.trailing_stop > self.stop_loss:
                return "trailing_stop"
        elif self.side == "short":
            if current_price >= self.stop_loss:
                return "stop_loss"
            if current_price <= self.take_profit:
                return "take_profit"
            if current_price >= self.trailing_stop and self.trailing_stop < self.stop_loss:
                return "trailing_stop"

        # Max hold duration
        if current_bar > 0 and self.entry_bar > 0 and current_bar - self.entry_bar >= Config.MAX_HOLD_BARS:
            return "max_duration"

        return None

    def unrealized_pnl(self, current_price: float) -> float:
        """Calculate unrealized profit/loss at the given price.

        Args:
            current_price: Current market price of the asset.

        Returns:
            Unrealized PnL in quote currency (USD). Positive means profit,
            negative means loss. Does not include fees.
        """
        if self.side == "long":
            return (current_price - self.entry_price) * self.quantity
        else:
            return (self.entry_price - current_price) * self.quantity


@dataclass
class TradeRecord:
    """Immutable record of a completed (closed) trade."""

    symbol: str
    side: str
    entry_price: float
    exit_price: float
    quantity: float
    pnl_gross: float  # Before fees
    pnl_net: float  # After fees
    fees_paid: float
    entry_time: datetime
    exit_time: datetime
    exit_reason: str
    strategy_name: str = ""
    hold_bars: int = 0


class RiskManager:
    """Portfolio-level risk manager with volatility-adaptive sizing and tiered drawdown protocol."""

    # Volatility targeting constants
    TARGET_ANNUAL_VOL = 0.15  # 15% annualized portfolio volatility target
    VOL_LOOKBACK = 20  # Bars for realized volatility calculation
    BARS_PER_YEAR = 8760  # 1h bars/year for annualization
    # Tiered drawdown thresholds
    DD_TIER1_PCT = 0.05  # -5%: reduce risk 25%
    DD_TIER2_PCT = 0.10  # -10%: reduce risk 50%, A-setups only
    DD_TIER3_PCT = 0.15  # -15%: halt trading 24h, review

    def __init__(self) -> None:
        """Initialize the risk manager with capital set to ``Config.INITIAL_CAPITAL``.

        Sets up empty position and trade-history lists, resets daily/total PnL
        counters, initializes volatility-adaptive state (rolling returns, peak
        capital, halt timer), and creates the internal threading lock used for
        atomic check-and-open operations.
        """
        self.capital = Config.INITIAL_CAPITAL
        self.positions: list[Position] = []
        self.trade_history: list[TradeRecord] = []
        self.daily_pnl: float = 0.0
        self.daily_pnl_date: date = date.today()
        self.total_pnl: float = 0.0
        self.total_fees: float = 0.0
        self.current_bar: int = 0
        # Volatility-adaptive state
        self._recent_returns: list[float] = []  # Rolling returns for vol calc
        self._peak_capital: float = Config.INITIAL_CAPITAL
        self._halt_until: datetime | None = None  # Trading halt timestamp
        self._position_correlations: dict[str, float] = {}  # Pair correlations
        # Thread safety: protects positions list and capital during check-and-open
        self._lock = threading.Lock()

    def _reset_daily_pnl_if_needed(self) -> None:
        """Reset the daily PnL counter when the calendar date has changed."""
        today = date.today()
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today

    def set_bar(self, bar: int) -> None:
        """Set the current bar counter used for max-hold-duration tracking.

        Args:
            bar: Monotonically increasing bar index (typically the agent's
                cycle count).
        """
        self.current_bar = bar

    def can_open_position(self, signal: str, confidence: float, symbol: str | None = None) -> tuple[bool, str]:
        """Check whether a new position may be opened.

        Thread-safe — acquires the internal lock to prevent TOCTOU race
        conditions. Cheap checks run first; the position loop runs last.

        Args:
            signal: Proposed signal — ``"BUY"`` or ``"SELL"``.
            confidence: Signal confidence in ``[0, 1]``. Must meet
                ``Config.MIN_CONFIDENCE`` or the position is blocked.
            symbol: Trading pair to check, e.g. ``"BTC/USDT"``. Falls back
                to ``Config.TRADING_PAIR`` when ``None``.

        Returns:
            Tuple ``(allowed, reason)`` where ``allowed`` is ``True`` when a
            position can be opened and ``reason`` is ``"OK"`` or a human-
            readable explanation for the block.
        """
        with self._lock:
            return self._can_open_position_unlocked(signal, confidence, symbol)

    def _can_open_position_unlocked(
        self, signal: str, confidence: float, symbol: str | None = None
    ) -> tuple[bool, str]:
        """Internal position-open check — must be called within the lock.

        Args:
            signal: Proposed signal — ``"BUY"`` or ``"SELL"``.
            confidence: Signal confidence in ``[0, 1]``.
            symbol: Trading pair symbol. Falls back to ``Config.TRADING_PAIR``.

        Returns:
            Tuple ``(allowed, reason)`` mirroring ``can_open_position``.
        """
        symbol = symbol or Config.TRADING_PAIR

        if confidence < Config.MIN_CONFIDENCE:
            return False, f"Confidence {confidence:.2%} below threshold {Config.MIN_CONFIDENCE:.2%}"

        if len(self.positions) >= Config.MAX_OPEN_POSITIONS:
            return False, f"Max positions reached ({Config.MAX_OPEN_POSITIONS})"

        self._reset_daily_pnl_if_needed()
        daily_loss_limit = Config.INITIAL_CAPITAL * Config.MAX_DAILY_LOSS_PCT
        if self.daily_pnl < -daily_loss_limit:
            return False, f"Daily loss limit hit (${self.daily_pnl:.2f})"

        target_side = "long" if signal == "BUY" else "short"
        opposite_side = "short" if signal == "BUY" else "long"
        for pos in self.positions:
            if pos.symbol == symbol:
                if pos.side == opposite_side:
                    return False, "Conflicting position already open"
                if pos.side == target_side:
                    return False, "Already have a position in this direction"

        return True, "OK"

    def atomic_check_and_open(
        self,
        signal: str,
        confidence: float,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str = "",
    ) -> tuple[bool, str, Position | None]:
        """Atomically check whether a position can be opened and open it.

        Combines ``can_open_position`` and ``open_position`` under a single
        lock acquisition to prevent TOCTOU races in multi-pair trading.

        Args:
            signal: Signal used for the eligibility check (``"BUY"`` /
                ``"SELL"``).
            confidence: Signal confidence in ``[0, 1]``.
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Position side — ``"long"`` or ``"short"``.
            entry_price: Price at which the position is entered.
            quantity: Asset quantity to trade.
            stop_loss: Fixed stop-loss price level.
            take_profit: Fixed take-profit price level.
            strategy_name: Name of the strategy that generated the signal.

        Returns:
            Tuple ``(success, reason, position_or_none)``.  ``success`` is
            ``True`` and ``position_or_none`` is the new ``Position`` when the
            trade was opened; ``success`` is ``False`` and ``position_or_none``
            is ``None`` when blocked.
        """
        with self._lock:
            can_open, reason = self._can_open_position_unlocked(signal, confidence, symbol)
            if not can_open:
                return False, reason, None
            pos = self._open_position_unlocked(
                symbol,
                side,
                entry_price,
                quantity,
                stop_loss,
                take_profit,
                strategy_name,
            )
            return True, "OK", pos

    def calculate_position_size(
        self,
        entry_price: float,
        fee_pct: float | None = None,
        confidence: float = 0.6,
        strategy_name: str = "",
        regime: str = "",
        atr_pct: float = 0.0,
        symbol: str = "",
    ) -> float:
        """Compute a volatility-adaptive position size (v9.0).

        Applies a six-layer sizing pipeline:
        1. Kelly fraction from strategy trade history.
        2. Volatility targeting (target 15% annualized portfolio vol).
        3. Confidence scaling (0.5× – 1.5× relative to a 0.7 baseline).
        4. Regime scaling (crash → 0.3×, trending → 1.3×).
        5. Tiered drawdown protocol (−5% → 0.75×, −10% → 0.5×, −15% → halt).
        6. Correlation adjustment (concentrated same-direction → penalty).

        The final percentage is clamped to ``[0.3%, 5%]`` of available capital.

        Args:
            entry_price: Price at which the position will be entered.
            fee_pct: One-way (entry-leg) fee as a decimal. The exit fee is
                charged separately at close time. Defaults to
                ``Config.FEE_PCT`` when ``None``.
            confidence: Signal confidence in ``[0, 1]``.
            strategy_name: Strategy name for per-strategy Kelly estimation.
            regime: Market regime string, e.g. ``"trending_up"``.
            atr_pct: ATR as a fraction of current price. Used by the
                volatility targeting layer and the correlation step.
            symbol: Trading pair symbol used by the correlation layer.

        Returns:
            Asset quantity to trade, rounded to 8 decimal places. Returns
            ``0.0`` if trading is currently halted by the drawdown protocol.
        """
        fee = fee_pct or Config.FEE_PCT
        available = self.capital

        # Check if trading is halted
        if self._halt_until and datetime.now() < self._halt_until:
            return 0.0

        # Layer 1: Kelly fraction from strategy track record
        kelly_pct = self._kelly_fraction(strategy_name)

        # Layer 2: Volatility targeting
        vol_mult = self._volatility_target_multiplier(atr_pct)

        # Layer 3: Confidence scaling
        confidence_mult = max(0.5, min(1.5, confidence / 0.7))

        # Layer 4: Regime scaling
        regime_mult = self._regime_position_mult(regime)

        # Layer 5: Tiered drawdown protocol
        drawdown_mult, dd_tier = self._tiered_drawdown_scaling()

        # Layer 6: Correlation adjustment
        corr_mult = self._correlation_adjustment(symbol)

        # Combine all multipliers
        position_pct = kelly_pct * vol_mult * confidence_mult * regime_mult * drawdown_mult * corr_mult

        # Clamp between 0.3% and 5% of capital
        position_pct = max(0.003, min(0.05, position_pct))

        risk_amount = available * position_pct
        effective_amount = risk_amount / (1 + fee)
        quantity = effective_amount / entry_price
        return round(quantity, 8)

    def _kelly_fraction(self, strategy_name: str = "") -> float:
        """Compute the quarter-Kelly position fraction from trade history.

        Formula: ``Kelly% = (W * R - L) / R``, then multiplied by ``0.25``
        for quarter-Kelly safety. Requires at least 10 historical trades for
        a reliable estimate.

        Args:
            strategy_name: Filter trades to this strategy. Uses all trades
                when empty.

        Returns:
            Quarter-Kelly fraction as a decimal (e.g. ``0.01`` for 1%).
            Returns ``Config.MAX_POSITION_PCT`` when fewer than 10 trades are
            available. Returns ``min(0.01, Config.MAX_POSITION_PCT)`` when
            there are no losing trades (avoids aggressive sizing from
            incomplete data).
        """
        # Get trades for this strategy (or all trades if no name)
        if strategy_name:
            trades = [t for t in self.trade_history if t.strategy_name == strategy_name]
        else:
            trades = list(self.trade_history)

        # Need at least 10 trades for reliable Kelly estimate
        if len(trades) < 10:
            return Config.MAX_POSITION_PCT  # Fall back to fixed

        wins = [t for t in trades if t.pnl_net > 0]
        losses = [t for t in trades if t.pnl_net <= 0]

        if not wins or not losses:
            return min(0.01, Config.MAX_POSITION_PCT)

        win_rate = len(wins) / len(trades)
        avg_win = sum(t.pnl_net for t in wins) / len(wins)
        avg_loss = abs(sum(t.pnl_net for t in losses) / len(losses))

        if avg_loss == 0:
            return Config.MAX_POSITION_PCT

        reward_risk = avg_win / avg_loss  # R
        kelly = (win_rate * reward_risk - (1 - win_rate)) / reward_risk

        # Quarter Kelly for safety (captures 75% of growth, 50% less drawdown)
        quarter_kelly = kelly * 0.25

        # Kelly can be negative (don't trade!) — clamp to small positive
        return max(0.005, quarter_kelly)

    def _regime_position_mult(self, regime: str) -> float:
        """Return the position-size multiplier for the given market regime.

        Args:
            regime: Market regime string, e.g. ``"trending_up"``,
                ``"ranging"``, ``"volatile"``, ``"breakout"``, ``"crash"``.

        Returns:
            Multiplier in the range ``[0.3, 1.3]``. Key values:
            ``trending_up`` → 1.3, ``trending_down`` → 1.1,
            ``breakout`` → 1.2, ``ranging`` → 0.7, ``volatile`` → 0.5,
            ``crash`` → 0.3. Returns ``1.0`` for unknown regimes.
        """
        regime_mults = {
            "trending_up": 1.3,
            "trending_down": 1.1,  # Slightly larger for shorts in trend
            "ranging": 0.7,
            "volatile": 0.5,
            "breakout": 1.2,
            "crash": 0.3,
        }
        return regime_mults.get(regime, 1.0)

    def _drawdown_scaling(self) -> float:
        """Return the drawdown multiplier (legacy shim for the backtester).

        Delegates to ``_tiered_drawdown_scaling`` and returns only the
        multiplier component. New code should call ``_tiered_drawdown_scaling``
        directly to also obtain the tier level.

        Returns:
            Position-size multiplier in ``[0.0, 1.0]``.
        """
        mult, _ = self._tiered_drawdown_scaling()
        return mult

    def _tiered_drawdown_scaling(self) -> tuple[float, int]:
        """Return the tiered drawdown multiplier and tier level.

        Implements a progressive risk-reduction protocol based on peak-to-
        current drawdown:

        - Tier 0 (DD ≤ 5%) :  1.00× — normal trading.
        - Tier 1 (DD > 5%) :  0.75× — reduce risk 25%.
        - Tier 2 (DD > 10%):  0.50× — reduce risk 50%, A-setups only.
        - Tier 3 (DD > 15%):  0.00× — halt trading for 24 hours.

        Returns:
            Tuple ``(multiplier, tier)`` where ``multiplier`` is the position-
            size scalar and ``tier`` is an integer from 0 to 3.
        """
        if self.capital <= 0:
            return 0.0, 3

        # Use peak capital for drawdown (not just initial)
        self._peak_capital = max(self._peak_capital, self.capital)
        current_dd = (self._peak_capital - self.capital) / self._peak_capital

        if current_dd > self.DD_TIER3_PCT:
            # Tier 3: Halt trading for 24 hours
            if self._halt_until is None or datetime.now() >= self._halt_until:
                from datetime import timedelta

                self._halt_until = datetime.now() + timedelta(hours=24)
                _log.warning(
                    "[Risk] TIER 3 HALT: DD=%.1f%% > %.0f%%. Trading halted until %s",
                    current_dd * 100,
                    self.DD_TIER3_PCT * 100,
                    self._halt_until.strftime("%H:%M"),
                )
            return 0.0, 3
        elif current_dd > self.DD_TIER2_PCT:
            return 0.50, 2
        elif current_dd > self.DD_TIER1_PCT:
            return 0.75, 1
        return 1.0, 0

    def _volatility_target_multiplier(self, atr_pct: float = 0.0) -> float:
        """Compute the volatility-targeting multiplier.

        Scales position size so that realized portfolio volatility converges
        toward ``TARGET_ANNUAL_VOL`` (15% annualized). The multiplier equals
        ``TARGET_ANNUAL_VOL / realized_vol``, clamped to ``[0.3, 2.0]``.

        Examples:
            - Realized vol 30%, target 15% → multiplier = 0.5 (halve size).
            - Realized vol 10%, target 15% → multiplier = 1.5 (increase size).

        Args:
            atr_pct: ATR as a fraction of the current price. Used as a proxy
                for realized volatility when provided. Falls back to the
                rolling return series when ``atr_pct <= 0``.

        Returns:
            Multiplier clamped to ``[0.3, 2.0]``. Returns ``1.0`` when
            insufficient data is available.
        """
        if atr_pct > 0:
            # ATR-based volatility estimate (annualized)
            realized_vol = atr_pct * np.sqrt(self.BARS_PER_YEAR)
        elif len(self._recent_returns) >= 10:
            # Use recent return series
            realized_vol = float(np.std(self._recent_returns)) * np.sqrt(self.BARS_PER_YEAR)
        else:
            return 1.0  # Not enough data yet

        if realized_vol <= 0:
            return 1.0

        vol_mult = self.TARGET_ANNUAL_VOL / realized_vol
        # Clamp between 0.3x and 2.0x
        return max(0.3, min(2.0, vol_mult))

    def _correlation_adjustment(self, symbol: str = "") -> float:
        """Reduce position size when open positions are highly correlated.

        Uses a simple proxy: same-direction positions (all long or all short)
        signal concentrated directional exposure and attract a penalty.

        Args:
            symbol: Incoming trading pair symbol (currently unused but
                reserved for future pair-specific correlation tables).

        Returns:
            Multiplier in ``(0, 1]``:
            - 0.50 — 3+ positions, all same direction.
            - 0.70 — 2+ positions, all same direction.
            - 0.85 — 2+ positions, mixed directions.
            - 1.00 — 0 or 1 position.
        """
        if not symbol or not self.positions:
            return 1.0

        # Count positions in same "asset class" (crypto pairs often correlated)
        n_open = len(self.positions)
        if n_open == 0:
            return 1.0

        # Simple correlation proxy: same-direction positions = higher correlation
        same_direction = sum(1 for p in self.positions if p.side == "long")
        all_same = same_direction == n_open or same_direction == 0

        if n_open >= 3 and all_same:
            return 0.5  # Heavy concentration penalty
        elif n_open >= 2 and all_same:
            return 0.7  # Moderate concentration
        elif n_open >= 2:
            return 0.85  # Some diversification benefit
        return 1.0

    def update_returns(self, bar_return: float) -> None:
        """Append a bar return to the rolling window used for volatility estimation.

        Maintains a rolling window of ``VOL_LOOKBACK * 2`` returns; older
        observations are discarded.

        Args:
            bar_return: Fractional return for the completed bar
                (e.g. ``0.005`` for +0.5%).
        """
        self._recent_returns.append(bar_return)
        # Keep rolling window
        if len(self._recent_returns) > self.VOL_LOOKBACK * 2:
            self._recent_returns = self._recent_returns[-self.VOL_LOOKBACK :]

    def get_risk_status(self) -> dict[str, Any]:
        """Return a detailed risk status snapshot including volatility metrics.

        Returns:
            Dict with keys: ``capital``, ``peak_capital``,
            ``current_drawdown_pct``, ``drawdown_tier``,
            ``realized_vol_annual`` (%), ``target_vol_annual`` (%),
            ``vol_multiplier``, ``is_halted``, ``halt_until`` (ISO-8601 or
            ``None``), ``open_positions``, ``total_trades``, ``total_pnl``,
            ``daily_pnl``, ``win_rate``.
        """
        _, dd_tier = self._tiered_drawdown_scaling()
        self._peak_capital = max(self._peak_capital, self.capital)
        current_dd = (self._peak_capital - self.capital) / self._peak_capital

        realized_vol = 0.0
        if len(self._recent_returns) >= 10:
            realized_vol = float(np.std(self._recent_returns)) * np.sqrt(self.BARS_PER_YEAR)

        return {
            "capital": round(self.capital, 2),
            "peak_capital": round(self._peak_capital, 2),
            "current_drawdown_pct": round(current_dd * 100, 2),
            "drawdown_tier": dd_tier,
            "realized_vol_annual": round(realized_vol * 100, 2),
            "target_vol_annual": self.TARGET_ANNUAL_VOL * 100,
            "vol_multiplier": round(self._volatility_target_multiplier(), 2),
            "is_halted": self._halt_until is not None and datetime.now() < self._halt_until,
            "halt_until": self._halt_until.isoformat() if self._halt_until else None,
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "win_rate": self._win_rate(),
        }

    def calculate_stop_take(
        self,
        entry_price: float,
        side: str,
        sl_pct: float | None = None,
        tp_pct: float | None = None,
        atr: float | None = None,
        regime: str = "",
    ) -> tuple[float, float]:
        """Compute regime-adaptive stop-loss and take-profit price levels (v8.0).

        Priority for base distances:
        1. ``sl_pct`` and ``tp_pct`` when both are provided.
        2. ATR-based: SL = 2 × ATR, TP = 3 × ATR when ``atr > 0``.
        3. ``Config.STOP_LOSS_PCT`` / ``Config.TAKE_PROFIT_PCT`` as fallback.

        Regime multipliers then widen or tighten both levels:
        - trending: wide TP to ride the trend.
        - ranging: tight both for quick scalps.
        - volatile/crash: tight SL to limit damage.

        If the calculated SL would self-trigger at entry, a 2% fallback is
        applied and a warning is logged.

        Args:
            entry_price: Price at which the position is entered.
            side: Position side — ``"long"`` or ``"short"``.
            sl_pct: Fixed SL distance as a fraction of ``entry_price``
                (e.g. ``0.02`` for 2%). Both SL and TP must be provided
                together to use this path.
            tp_pct: Fixed TP distance as a fraction of ``entry_price``.
            atr: Absolute ATR value in price units. Used when ``sl_pct``/
                ``tp_pct`` are not provided.
            regime: Market regime string, e.g. ``"trending_up"``.

        Returns:
            Tuple ``(stop_loss, take_profit)`` rounded to 2 decimal places.
        """
        if sl_pct and tp_pct:
            sl_distance = entry_price * sl_pct
            tp_distance = entry_price * tp_pct
        elif atr and atr > 0:
            sl_distance = 2 * atr
            tp_distance = 3 * atr
        else:
            sl_distance = entry_price * Config.STOP_LOSS_PCT
            tp_distance = entry_price * Config.TAKE_PROFIT_PCT

        # v8.0: Regime-adaptive SL/TP multipliers
        sl_mult, tp_mult = self._regime_sltp_multipliers(regime)
        sl_distance *= sl_mult
        tp_distance *= tp_mult

        if side == "long":
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + tp_distance
            if stop_loss >= entry_price:
                _log.warning("SL >= entry for long (SL=%.2f, entry=%.2f); using 2%% fallback", stop_loss, entry_price)
                stop_loss = entry_price * 0.98
            if take_profit <= entry_price:
                _log.warning("TP <= entry for long (TP=%.2f, entry=%.2f); using 3%% fallback", take_profit, entry_price)
                take_profit = entry_price * 1.03
        else:
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - tp_distance
            if stop_loss <= entry_price:
                _log.warning("SL <= entry for short (SL=%.2f, entry=%.2f); using 2%% fallback", stop_loss, entry_price)
                stop_loss = entry_price * 1.02
            if take_profit >= entry_price:
                _log.warning("TP >= entry for short (TP=%.2f, entry=%.2f); using 3%% fallback", take_profit, entry_price)
                take_profit = entry_price * 0.97

        return round(stop_loss, 2), round(take_profit, 2)

    def _regime_sltp_multipliers(self, regime: str) -> tuple[float, float]:
        """Return ``(SL multiplier, TP multiplier)`` for the given market regime.

        Args:
            regime: Market regime string, e.g. ``"trending_up"``,
                ``"ranging"``, ``"volatile"``, ``"crash"``.

        Returns:
            Tuple ``(sl_mult, tp_mult)`` where:
            - trending_up/down: ``(1.0, 1.5 / 1.4)`` — wide TP, normal SL.
            - ranging: ``(0.8, 0.7)`` — tight both for quick scalps.
            - volatile: ``(0.7, 1.3)`` — tight SL, wide TP.
            - high_volatility: ``(0.7, 1.3)`` — same as ``volatile``.
            - breakout: ``(1.1, 1.8)`` — slightly wider SL, much wider TP.
            - crash: ``(0.5, 0.6)`` — very tight everything.
            - unknown: ``(1.0, 1.0)`` — no adjustment.
        """
        regime_params = {
            "trending_up": (1.0, 1.5),  # Normal SL, 50% wider TP
            "trending_down": (1.0, 1.4),  # Normal SL, 40% wider TP (for shorts)
            "ranging": (0.8, 0.7),  # 20% tighter SL, 30% tighter TP (scalp mode)
            "volatile": (0.7, 1.3),  # 30% tighter SL, 30% wider TP
            "high_volatility": (0.7, 1.3),  # Same as volatile
            "breakout": (1.1, 1.8),  # Slightly wider SL, much wider TP
            "crash": (0.5, 0.6),  # Very tight everything
        }
        return regime_params.get(regime, (1.0, 1.0))

    def open_position(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str = "",
    ) -> Position:
        """Record a new position with trailing stop initialized. Thread-safe.

        Deducts entry cost plus fee from ``self.capital`` and appends the new
        ``Position`` to ``self.positions``.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Position side — ``"long"`` or ``"short"``.
            entry_price: Price at which the position was filled.
            quantity: Asset quantity held.
            stop_loss: Fixed stop-loss price level.
            take_profit: Fixed take-profit price level.
            strategy_name: Name of the originating strategy.

        Returns:
            The newly created ``Position`` object with trailing stop
            pre-initialized from ``Config.TRAILING_STOP_PCT``.
        """
        with self._lock:
            return self._open_position_unlocked(
                symbol,
                side,
                entry_price,
                quantity,
                stop_loss,
                take_profit,
                strategy_name,
            )

    def _open_position_unlocked(
        self,
        symbol: str,
        side: str,
        entry_price: float,
        quantity: float,
        stop_loss: float,
        take_profit: float,
        strategy_name: str = "",
    ) -> Position:
        """Internal position-open logic — must be called within the lock.

        Args:
            symbol: Trading pair symbol.
            side: Position side — ``"long"`` or ``"short"``.
            entry_price: Fill price.
            quantity: Asset quantity.
            stop_loss: Fixed stop-loss price level.
            take_profit: Fixed take-profit price level.
            strategy_name: Originating strategy name.

        Returns:
            The newly created ``Position``.
        """
        fee = self._charge_fee(entry_price, quantity)

        pos = Position(
            symbol=symbol,
            side=side,
            entry_price=entry_price,
            quantity=quantity,
            entry_time=datetime.now(),
            stop_loss=stop_loss,
            take_profit=take_profit,
            strategy_name=strategy_name,
            entry_bar=self.current_bar,
        )
        self.positions.append(pos)
        self.capital -= entry_price * quantity + fee

        _log.info(
            "[Risk] OPENED %s %s: qty=%.6f @ $%,.2f | SL=$%,.2f TP=$%,.2f Trail=$%,.2f | Fee=$%.4f",
            side.upper(),
            symbol,
            quantity,
            entry_price,
            stop_loss,
            take_profit,
            pos.trailing_stop,
            fee,
        )
        return pos

    def close_position(self, position: Position, exit_price: float, reason: str) -> TradeRecord:
        """Close an open position with fee accounting. Thread-safe.

        Calculates gross and net PnL, updates capital and cumulative counters,
        removes the position from ``self.positions``, and appends a
        ``TradeRecord`` to ``self.trade_history``.

        Args:
            position: The ``Position`` object to close.
            exit_price: Price at which the position was exited.
            reason: Exit reason string, e.g. ``"stop_loss"``,
                ``"take_profit"``, ``"trailing_stop"``, ``"max_duration"``,
                or ``"force_close"``.

        Returns:
            Completed ``TradeRecord`` with full PnL, fee, and timing data.
        """
        with self._lock:
            return self._close_position_unlocked(position, exit_price, reason)

    def _close_position_unlocked(self, position: Position, exit_price: float, reason: str) -> TradeRecord:
        """Internal position-close logic — must be called within the lock.

        Args:
            position: The ``Position`` to close.
            exit_price: Exit fill price.
            reason: Human-readable exit reason.

        Returns:
            Completed ``TradeRecord``.
        """
        fee = self._charge_fee(exit_price, position.quantity)
        pnl_gross = position.unrealized_pnl(exit_price)
        pnl_net = pnl_gross - fee  # Exit fee only (entry already deducted)

        self.daily_pnl += pnl_net
        self.total_pnl += pnl_net
        self.capital += exit_price * position.quantity - fee

        hold_bars = self.current_bar - position.entry_bar if position.entry_bar > 0 else 0

        record = TradeRecord(
            symbol=position.symbol,
            side=position.side,
            entry_price=position.entry_price,
            exit_price=exit_price,
            quantity=position.quantity,
            pnl_gross=pnl_gross,
            pnl_net=pnl_net,
            fees_paid=fee,
            entry_time=position.entry_time,
            exit_time=datetime.now(),
            exit_reason=reason,
            strategy_name=position.strategy_name,
            hold_bars=hold_bars,
        )
        self.trade_history.append(record)
        self.positions.remove(position)

        emoji = "+" if pnl_net >= 0 else "-"
        _log.info(
            "[Risk] [%s] CLOSED %s %s: PnL=$%,.2f (gross $%,.2f) | %s | held %d bars",
            emoji,
            position.side.upper(),
            position.symbol,
            pnl_net,
            pnl_gross,
            reason,
            hold_bars,
        )
        return record

    def _charge_fee(self, price: float, quantity: float) -> float:
        """Compute and record a one-way fee.

        Args:
            price: Fill price for this leg of the trade.
            quantity: Asset quantity traded.

        Returns:
            Fee amount in quote currency (USD).
        """
        fee = price * quantity * Config.FEE_PCT
        self.total_fees += fee
        return fee

    def check_positions(
        self,
        current_price: float,
        current_high: float | None = None,
        current_low: float | None = None,
        symbol: str | None = None,
        atr_pct: float | None = None,
    ) -> list[TradeRecord]:
        """Update trailing stops and close any positions that hit an exit condition.

        For each eligible position the method:
        1. Updates the trailing stop using intra-bar high/low.
        2. Checks and applies the breakeven stop.
        3. Evaluates all exit conditions (SL, TP, trailing stop, max duration).
        4. Closes the position if an exit is triggered.

        Args:
            current_price: Latest close price used for exit evaluation and
                unrealized PnL.
            current_high: Intra-bar high price for trailing stop ratchet.
                Defaults to ``current_price`` when ``None``.
            current_low: Intra-bar low price for trailing stop ratchet.
                Defaults to ``current_price`` when ``None``.
            symbol: When provided, only positions matching this trading pair
                are checked — prevents cross-pair price contamination in
                multi-pair mode.
            atr_pct: ATR as a fraction of the current price. When provided,
                trailing stop distance = ``Config.ATR_TRAILING_MULT × atr_pct``
                instead of the fixed ``Config.TRAILING_STOP_PCT``.

        Returns:
            List of ``TradeRecord`` objects for positions closed this call.
            Empty list when no exits were triggered.
        """
        high = current_high or current_price
        low = current_low or current_price

        closed = []
        for pos in list(self.positions):
            if symbol and pos.symbol != symbol:
                continue
            # ATR-adaptive trailing stop update (v9.1)
            pos.update_trailing_stop(high, low, atr_pct=atr_pct)
            # Breakeven stop check (v9.1)
            pos.check_breakeven(current_price, Config.FEE_PCT)

            exit_reason = pos.check_exit(current_price, self.current_bar)
            if exit_reason:
                record = self.close_position(pos, current_price, exit_reason)
                closed.append(record)
        return closed

    def get_summary(self) -> dict[str, Any]:
        """Return a compact snapshot of portfolio state.

        Returns:
            Dict with keys: ``capital``, ``open_positions``, ``total_trades``,
            ``total_pnl``, ``daily_pnl``, ``total_fees``, ``win_rate``.
        """
        return {
            "capital": round(self.capital, 2),
            "open_positions": len(self.positions),
            "total_trades": len(self.trade_history),
            "total_pnl": round(self.total_pnl, 2),
            "daily_pnl": round(self.daily_pnl, 2),
            "total_fees": round(self.total_fees, 2),
            "win_rate": self._win_rate(),
        }

    def _win_rate(self) -> float:
        """Compute the historical win rate from completed trades.

        Returns:
            Fraction of trades with positive net PnL, in ``[0, 1]``, rounded
            to 4 decimal places. Returns ``0.0`` when no trades exist.
        """
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.pnl_net > 0)
        return round(wins / len(self.trade_history), 4)

    # --- State persistence ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize portfolio state for persistence.

        Returns:
            JSON-serializable dict containing capital, PnL counters, open
            positions (with all trailing-stop fields), and trade count.
            Trade history records are not persisted (only the count).
        """
        return {
            "capital": self.capital,
            "total_pnl": self.total_pnl,
            "total_fees": self.total_fees,
            "daily_pnl": self.daily_pnl,
            "daily_pnl_date": str(self.daily_pnl_date),
            "current_bar": self.current_bar,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "entry_time": p.entry_time.isoformat(),
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                    "highest_price": p.highest_price,
                    "lowest_price": p.lowest_price,
                    "strategy_name": p.strategy_name,
                    "entry_bar": p.entry_bar,
                    "breakeven_triggered": p.breakeven_triggered,
                }
                for p in self.positions
            ],
            "trade_count": len(self.trade_history),
        }

    def from_dict(self, data: dict[str, Any]) -> None:
        """Restore portfolio state from a previously serialized dict.

        Reconstructs all ``Position`` objects from their stored fields.
        Missing optional fields fall back to safe defaults for forward
        compatibility with older state files.

        Args:
            data: Dict as produced by ``to_dict()``.
        """
        self.capital = data["capital"]
        self.total_pnl = data["total_pnl"]
        self.total_fees = data.get("total_fees", 0)
        self.daily_pnl = data["daily_pnl"]
        self.daily_pnl_date = date.fromisoformat(data["daily_pnl_date"])
        self.current_bar = data.get("current_bar", 0)
        self.positions = []
        for p in data.get("positions", []):
            pos = Position(
                symbol=p["symbol"],
                side=p["side"],
                entry_price=p["entry_price"],
                quantity=p["quantity"],
                entry_time=datetime.fromisoformat(p["entry_time"]),
                stop_loss=p["stop_loss"],
                take_profit=p["take_profit"],
                strategy_name=p.get("strategy_name", ""),
                trailing_stop=p.get("trailing_stop", 0),
                highest_price=p.get("highest_price", p["entry_price"]),
                lowest_price=p.get("lowest_price", p["entry_price"]),
                entry_bar=p.get("entry_bar", 0),
                breakeven_triggered=p.get("breakeven_triggered", False),
            )
            self.positions.append(pos)
