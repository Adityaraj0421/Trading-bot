"""
Risk Management module v4.1
============================
v4.1: Thread-safe position management with atomic check-and-open.
v4.0: Volatility-adaptive position sizing, tiered drawdown protocol,
      correlation-adjusted sizing, volatility targeting (15% annual).
v3.0: Trailing stop-loss, transaction cost awareness, max hold duration,
      position timeout, enhanced portfolio tracking.
"""

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
        """
        Update trailing stop based on new price extremes.

        When atr_pct is provided, trailing distance = ATR_TRAILING_MULT × atr_pct
        (adaptive to current volatility). Falls back to TRAILING_STOP_PCT otherwise.
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
        """Check all exit conditions. Returns reason or None."""
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
        """Calculate unrealized profit/loss at the given price."""
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
        today = date.today()
        if today != self.daily_pnl_date:
            self.daily_pnl = 0.0
            self.daily_pnl_date = today

    def set_bar(self, bar: int) -> None:
        """Set current bar number (for max hold tracking)."""
        self.current_bar = bar

    def can_open_position(self, signal: str, confidence: float, symbol: str | None = None) -> tuple[bool, str]:
        """
        Check if a new position is allowed.
        Thread-safe: acquires lock to prevent TOCTOU race conditions.
        OPTIMIZED: cheap checks first, single loop for position checks.
        """
        with self._lock:
            return self._can_open_position_unlocked(signal, confidence, symbol)

    def _can_open_position_unlocked(
        self, signal: str, confidence: float, symbol: str | None = None
    ) -> tuple[bool, str]:
        """Internal check without lock (called within locked context)."""
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
    ) -> tuple[bool, str, "Position | None"]:
        """
        Atomically check if position can be opened AND open it.
        Prevents TOCTOU race condition in multi-pair trading.
        Returns (success, reason, position_or_none).
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
        """
        v9.0: Volatility-adaptive position sizing.
        Layers: Kelly fraction -> volatility targeting -> confidence scaling
        -> regime scaling -> tiered drawdown protocol -> correlation adjustment.

        New in v9.0:
        - Volatility targeting: scales position to target 15% annual portfolio vol
        - Tiered drawdown: progressive risk reduction (-5%, -10%, -15%)
        - Correlation-adjusted: reduces size when positions are correlated
        - ATR-percentile scaling: high volatility = smaller positions
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
        """
        Compute quarter-Kelly fraction from trade history.
        Kelly% = (W * R - L) / R, then multiply by 0.25 for quarter-Kelly.
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
        """Regime-adaptive position multiplier."""
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
        """Legacy drawdown scaling (used by backtester). See _tiered_drawdown_scaling."""
        _, _ = self._tiered_drawdown_scaling()
        mult, _ = self._tiered_drawdown_scaling()
        return mult

    def _tiered_drawdown_scaling(self) -> tuple[float, int]:
        """
        Tiered drawdown protocol with progressive risk reduction.
        Returns (multiplier, tier_level).

        Tier 0: Normal        -> 1.0x
        Tier 1: DD > 5%       -> 0.75x (reduce risk 25%)
        Tier 2: DD > 10%      -> 0.50x (reduce risk 50%, A-setups only)
        Tier 3: DD > 15%      -> 0.0x  (halt trading for 24h)
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
        """
        Volatility targeting: scale position to target constant portfolio vol.
        target_vol / realized_vol = position_multiplier

        If realized vol is 30% and target is 15%, we halve position size.
        If realized vol is 10% and target is 15%, we increase by 1.5x.
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
        """
        Reduce position size when open positions are correlated.
        E.g., being long BTC and long ETH is concentrated crypto exposure.
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
        """Track recent returns for volatility calculation."""
        self._recent_returns.append(bar_return)
        # Keep rolling window
        if len(self._recent_returns) > self.VOL_LOOKBACK * 2:
            self._recent_returns = self._recent_returns[-self.VOL_LOOKBACK :]

    def get_risk_status(self) -> dict[str, Any]:
        """Get detailed risk status including volatility metrics."""
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
        """
        v8.0: Regime-adaptive stop-loss and take-profit levels.
        In trending regimes, widens TP to ride the trend.
        In volatile/crash regimes, tightens SL to limit damage.
        In ranging regimes, tightens both for quick scalps.
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
        """
        Return (SL multiplier, TP multiplier) based on market regime.
        Trending: wide TP, normal SL (ride the wave)
        Ranging: tight both (quick in and out)
        Volatile: tight SL, wide TP (protect capital, catch spikes)
        Crash: very tight SL (preserve capital at all costs)
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
        """Record a new position with trailing stop initialized. Thread-safe."""
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
        """Internal open (called within locked context)."""
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
        """Close position with fee accounting. Thread-safe."""
        with self._lock:
            return self._close_position_unlocked(position, exit_price, reason)

    def _close_position_unlocked(self, position: Position, exit_price: float, reason: str) -> TradeRecord:
        """Internal close (called within locked context)."""
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
        """
        Check positions for exits: update trailing stops, check breakeven, then check exits.
        Accepts high/low for intra-bar trailing stop updates.

        Args:
            symbol: If provided, only check positions for this trading pair.
                    This prevents cross-pair price contamination in multi-pair mode.
            atr_pct: ATR as % of price for this bar. When provided, trailing stop
                     uses ATR_TRAILING_MULT × atr_pct instead of fixed TRAILING_STOP_PCT.
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
        """Return a compact summary of capital, positions, PnL, and win rate."""
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
        if not self.trade_history:
            return 0.0
        wins = sum(1 for t in self.trade_history if t.pnl_net > 0)
        return round(wins / len(self.trade_history), 4)

    # --- State persistence ---
    def to_dict(self) -> dict[str, Any]:
        """Serialize state for persistence."""
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
        """Restore state from persistence."""
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
