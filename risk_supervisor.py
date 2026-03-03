"""RiskSupervisor — cross-cutting kill switch governor.

Watches four conditions and calls disable_new_trades() when any is breached:
  1. Daily drawdown > daily_drawdown_limit (default 3%)
  2. Consecutive losses > consecutive_loss_limit (default 4)
  3. ATR spike > atr_sigma_threshold σ above rolling mean (default 3σ)
  4. API error rate > api_error_rate_threshold (default 50%)

Single power: disable_new_trades() / enable_trades().
Does NOT adjust positions, scores, sizes, or context beyond risk_mode.

Re-enables automatically after cooldown_minutes (default 120 = 2h),
or immediately via enable_trades() (Telegram override).
"""

from __future__ import annotations

import logging
from collections import deque
from datetime import UTC, datetime

import numpy as np

_log = logging.getLogger(__name__)

# Thresholds
_DAILY_DRAWDOWN_LIMIT: float = 0.03    # 3 % of daily starting equity
_CONSECUTIVE_LOSS_LIMIT: int = 4       # Number of consecutive losses
_ATR_SIGMA_THRESHOLD: float = 3.0     # σ above rolling mean = shock
_API_ERROR_RATE_THRESHOLD: float = 0.5  # 50 % errors over window
_COOLDOWN_MINUTES: int = 120           # 2h default auto-re-enable

_MIN_ATR_HISTORY: int = 20             # Minimum ATR samples for σ calc
_MIN_API_CALLS: int = 10               # Minimum API calls before error rate fires


class RiskSupervisor:
    """Kill-switch governor for the trading system.

    Instantiate once at system startup and pass to ContextEngine and the main
    agent loop. Call the on_* event methods as events occur. Check
    ``is_trading_enabled()`` before placing new orders.

    Args:
        context_engine: ContextEngine instance whose risk_mode will be updated
            when trades are disabled/re-enabled. Pass None to omit this
            integration (useful in unit tests).
        daily_drawdown_limit: Fraction of daily starting equity; e.g. 0.03 = 3%.
        consecutive_loss_limit: Number of consecutive losses before kill.
        atr_sigma_threshold: ATR spike threshold in σ above rolling mean.
        api_error_rate_threshold: API error fraction that triggers kill.
        cooldown_minutes: Minutes before automatic re-enable after a kill.
    """

    def __init__(
        self,
        context_engine: object | None = None,  # ContextEngine; typed as object to avoid circular import
        daily_drawdown_limit: float = _DAILY_DRAWDOWN_LIMIT,
        consecutive_loss_limit: int = _CONSECUTIVE_LOSS_LIMIT,
        atr_sigma_threshold: float = _ATR_SIGMA_THRESHOLD,
        api_error_rate_threshold: float = _API_ERROR_RATE_THRESHOLD,
        cooldown_minutes: int = _COOLDOWN_MINUTES,
    ) -> None:
        self._context_engine = context_engine
        self._daily_drawdown_limit = daily_drawdown_limit
        self._consecutive_loss_limit = consecutive_loss_limit
        self._atr_sigma_threshold = atr_sigma_threshold
        self._api_error_rate_threshold = api_error_rate_threshold
        self._cooldown_minutes = cooldown_minutes

        # Mutable state
        self._trading_enabled: bool = True
        self._disabled_at: datetime | None = None
        self._disable_reason: str = ""

        # Daily tracking (reset by reset_daily())
        self._daily_pnl_pct: float = 0.0
        self._consecutive_losses: int = 0

        # ATR history (rolling, for σ calculation)
        self._atr_history: deque[float] = deque(maxlen=200)

        # API error tracking
        self._api_errors: int = 0
        self._api_calls: int = 0

    # ------------------------------------------------------------------
    # Event handlers — call these as events occur in the main loop
    # ------------------------------------------------------------------

    def on_trade_result(self, pnl_pct: float) -> None:
        """Update after each completed trade.

        Args:
            pnl_pct: Trade PnL as a fraction of position size (e.g. -0.015 = -1.5% loss).
                Positive = profit, negative = loss.
        """
        self._daily_pnl_pct += pnl_pct
        if pnl_pct < 0:
            self._consecutive_losses += 1
        else:
            self._consecutive_losses = 0

        if self._daily_pnl_pct < -abs(self._daily_drawdown_limit):
            self.disable_new_trades("daily_drawdown_exceeded")
        elif self._consecutive_losses > self._consecutive_loss_limit:
            self.disable_new_trades("consecutive_losses_exceeded")

    def on_atr_update(self, current_atr: float) -> None:
        """Update rolling ATR and check for volatility shock.

        Args:
            current_atr: Latest ATR value (same unit as price; not normalised).
        """
        self._atr_history.append(current_atr)
        if len(self._atr_history) < _MIN_ATR_HISTORY:
            return
        arr = np.array(self._atr_history, dtype=float)
        mean = float(arr.mean())
        std = float(arr.std())
        if std > 0 and current_atr > mean + self._atr_sigma_threshold * std:
            self.disable_new_trades("atr_spike")

    def on_api_error(self) -> None:
        """Record an API/exchange error. Disables trading if error rate too high."""
        self._api_errors += 1
        self._api_calls += 1
        self._check_api_error_rate()

    def on_api_success(self) -> None:
        """Record a successful API call. Also re-checks error rate."""
        self._api_calls += 1
        self._check_api_error_rate()

    def _check_api_error_rate(self) -> None:
        """Disable trading if error rate exceeds threshold over the minimum window."""
        if self._api_calls >= _MIN_API_CALLS:
            error_rate = self._api_errors / self._api_calls
            if error_rate >= self._api_error_rate_threshold:
                self.disable_new_trades("api_error_rate_exceeded")

    def reset_daily(self) -> None:
        """Reset daily PnL and consecutive-loss counters.

        Call this once at the start of each UTC trading day (midnight UTC).
        Does NOT re-enable trading — that happens via cooldown or manual override.
        """
        self._daily_pnl_pct = 0.0
        self._consecutive_losses = 0
        _log.info("RiskSupervisor: daily counters reset")

    def reset_api_counters(self) -> None:
        """Reset API error counters (e.g. after reconnecting to exchange)."""
        self._api_errors = 0
        self._api_calls = 0

    # ------------------------------------------------------------------
    # Kill switch
    # ------------------------------------------------------------------

    def disable_new_trades(self, reason: str) -> None:
        """Disable new trade entry and set context to defensive mode.

        Idempotent — calling again while already disabled updates the reason
        and resets the cooldown timer.

        Args:
            reason: Human-readable reason string for logging.
        """
        if not self._trading_enabled:
            # Already disabled — update reason and reset timer
            self._disable_reason = reason
            self._disabled_at = datetime.now(UTC)
            _log.warning("RiskSupervisor: still disabled, reason updated to %r", reason)
            return

        self._trading_enabled = False
        self._disabled_at = datetime.now(UTC)
        self._disable_reason = reason
        _log.warning("RiskSupervisor: trading DISABLED — %s", reason)

        if self._context_engine is not None:
            self._context_engine.set_risk_mode("defensive")

    def enable_trades(self) -> None:
        """Re-enable trading and restore normal risk mode.

        Called automatically after cooldown, or manually via Telegram override.
        """
        if self._trading_enabled:
            return
        self._trading_enabled = True
        self._disabled_at = None
        self._disable_reason = ""
        _log.info("RiskSupervisor: trading RE-ENABLED")

        if self._context_engine is not None:
            self._context_engine.set_risk_mode("normal")

    # ------------------------------------------------------------------
    # State queries
    # ------------------------------------------------------------------

    def is_trading_enabled(self) -> bool:
        """Return True if new trades are permitted.

        Checks cooldown expiry on each call and auto-re-enables if elapsed.

        Returns:
            True if the kill switch is off, False if kill switch is active.
        """
        if not self._trading_enabled and self._disabled_at is not None:
            elapsed_minutes = (datetime.now(UTC) - self._disabled_at).total_seconds() / 60
            if elapsed_minutes >= self._cooldown_minutes:
                _log.info(
                    "RiskSupervisor: cooldown elapsed (%.0f min), auto-re-enabling",
                    elapsed_minutes,
                )
                self.enable_trades()
        return self._trading_enabled

    @property
    def disable_reason(self) -> str:
        """Last reason trading was disabled (empty string when enabled)."""
        return self._disable_reason

    @property
    def daily_pnl_pct(self) -> float:
        """Cumulative daily PnL as a fraction (e.g. -0.02 = -2%)."""
        return self._daily_pnl_pct

    @property
    def consecutive_losses(self) -> int:
        """Current streak of consecutive losing trades."""
        return self._consecutive_losses
