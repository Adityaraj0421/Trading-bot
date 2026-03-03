# tests/test_risk_supervisor.py
"""Tests for RiskSupervisor kill-switch governor."""

from datetime import UTC, datetime, timedelta
from unittest.mock import MagicMock

from risk_supervisor import RiskSupervisor


class TestRiskSupervisor:
    # ------------------------------------------------------------------
    # Initial state
    # ------------------------------------------------------------------

    def test_trading_enabled_by_default(self):
        sup = RiskSupervisor()
        assert sup.is_trading_enabled() is True

    def test_disable_reason_empty_initially(self):
        sup = RiskSupervisor()
        assert sup.disable_reason == ""

    # ------------------------------------------------------------------
    # Daily drawdown trigger
    # ------------------------------------------------------------------

    def test_disables_on_daily_drawdown_exceeded(self):
        sup = RiskSupervisor(daily_drawdown_limit=0.03)
        sup.on_trade_result(-0.035)  # exceeds 3 %
        assert sup.is_trading_enabled() is False
        assert "drawdown" in sup.disable_reason

    def test_does_not_disable_on_drawdown_within_limit(self):
        sup = RiskSupervisor(daily_drawdown_limit=0.03)
        sup.on_trade_result(-0.02)   # within 3 %
        assert sup.is_trading_enabled() is True

    def test_cumulative_drawdown_triggers_kill(self):
        """Two small losses can sum to exceed the limit."""
        sup = RiskSupervisor(daily_drawdown_limit=0.03)
        sup.on_trade_result(-0.02)
        sup.on_trade_result(-0.015)  # cumulative = -3.5%
        assert sup.is_trading_enabled() is False

    # ------------------------------------------------------------------
    # Consecutive losses trigger
    # ------------------------------------------------------------------

    def test_disables_on_consecutive_losses_exceeded(self):
        sup = RiskSupervisor(consecutive_loss_limit=4)
        for _ in range(5):
            sup.on_trade_result(-0.001)
        assert sup.is_trading_enabled() is False
        assert "consecutive_losses" in sup.disable_reason

    def test_win_resets_consecutive_loss_counter(self):
        sup = RiskSupervisor(consecutive_loss_limit=4)
        for _ in range(3):
            sup.on_trade_result(-0.001)
        sup.on_trade_result(0.01)  # winning trade
        assert sup.consecutive_losses == 0
        assert sup.is_trading_enabled() is True

    def test_exactly_at_limit_does_not_disable(self):
        """Limit of 4 means 5 consecutive losses trigger, not 4."""
        sup = RiskSupervisor(consecutive_loss_limit=4)
        for _ in range(4):
            sup.on_trade_result(-0.001)
        assert sup.is_trading_enabled() is True  # 4 == limit, not > limit

    # ------------------------------------------------------------------
    # ATR spike trigger
    # ------------------------------------------------------------------

    def test_disables_on_atr_spike(self):
        sup = RiskSupervisor(atr_sigma_threshold=3.0)
        # Feed 20 low ATR values (mean ≈ 100, std ≈ 0)
        for _ in range(20):
            sup.on_atr_update(100.0)
        # Spike: 100 + very large value creates std, but needs to be 3σ above mean
        # Feed 20 varied values first to get a real std
        sup2 = RiskSupervisor(atr_sigma_threshold=3.0)
        baseline = [100.0 + 2.0 * (i % 5) for i in range(20)]
        for v in baseline:
            sup2.on_atr_update(v)
        # Spike: mean ≈ 104, std ≈ 3.2, threshold ≈ 104 + 3*3.2 = 113.6
        sup2.on_atr_update(200.0)  # well above threshold
        assert sup2.is_trading_enabled() is False
        assert "atr_spike" in sup2.disable_reason

    def test_no_disable_before_min_atr_history(self):
        sup = RiskSupervisor()
        for _ in range(5):   # less than _MIN_ATR_HISTORY = 20
            sup.on_atr_update(1000.0)
        assert sup.is_trading_enabled() is True

    # ------------------------------------------------------------------
    # API error rate trigger
    # ------------------------------------------------------------------

    def test_disables_on_api_error_rate_exceeded(self):
        sup = RiskSupervisor(api_error_rate_threshold=0.5)
        # Need at least 10 calls to trigger; send 6 errors + 4 success = 60% errors
        for _ in range(6):
            sup.on_api_error()
        for _ in range(4):
            sup.on_api_success()
        assert sup.is_trading_enabled() is False
        assert "api_error_rate" in sup.disable_reason

    def test_no_disable_below_min_api_calls(self):
        sup = RiskSupervisor(api_error_rate_threshold=0.5)
        for _ in range(9):   # less than _MIN_API_CALLS = 10
            sup.on_api_error()
        assert sup.is_trading_enabled() is True

    # ------------------------------------------------------------------
    # Cooldown auto-re-enable
    # ------------------------------------------------------------------

    def test_auto_reenables_after_cooldown(self):
        sup = RiskSupervisor(cooldown_minutes=0)  # instant cooldown for test
        sup.disable_new_trades("test")
        # Force disabled_at to the past
        sup._disabled_at = datetime.now(UTC) - timedelta(minutes=1)
        assert sup.is_trading_enabled() is True  # auto-re-enabled

    def test_does_not_reenable_before_cooldown(self):
        sup = RiskSupervisor(cooldown_minutes=120)
        sup.disable_new_trades("test")
        # disabled_at is just now, cooldown hasn't elapsed
        assert sup.is_trading_enabled() is False

    # ------------------------------------------------------------------
    # Manual override
    # ------------------------------------------------------------------

    def test_manual_enable_overrides_kill(self):
        sup = RiskSupervisor()
        sup.disable_new_trades("test")
        assert sup.is_trading_enabled() is False
        sup.enable_trades()
        assert sup.is_trading_enabled() is True
        assert sup.disable_reason == ""

    # ------------------------------------------------------------------
    # ContextEngine integration
    # ------------------------------------------------------------------

    def test_context_engine_set_defensive_on_disable(self):
        mock_ce = MagicMock()
        sup = RiskSupervisor(context_engine=mock_ce)
        sup.disable_new_trades("test")
        mock_ce.set_risk_mode.assert_called_once_with("defensive")

    def test_context_engine_set_normal_on_enable(self):
        mock_ce = MagicMock()
        sup = RiskSupervisor(context_engine=mock_ce)
        sup.disable_new_trades("test")
        sup.enable_trades()
        mock_ce.set_risk_mode.assert_called_with("normal")

    # ------------------------------------------------------------------
    # Daily reset
    # ------------------------------------------------------------------

    def test_reset_daily_clears_pnl_and_losses(self):
        sup = RiskSupervisor(daily_drawdown_limit=0.10, consecutive_loss_limit=10)
        sup.on_trade_result(-0.05)
        sup.on_trade_result(-0.02)
        assert sup.daily_pnl_pct < 0
        assert sup.consecutive_losses == 2
        sup.reset_daily()
        assert sup.daily_pnl_pct == 0.0
        assert sup.consecutive_losses == 0

    def test_reset_daily_does_not_reenable(self):
        """reset_daily() only clears counters — does NOT re-enable if killed."""
        sup = RiskSupervisor(daily_drawdown_limit=0.03)
        sup.on_trade_result(-0.05)
        assert sup.is_trading_enabled() is False
        sup.reset_daily()
        assert sup.is_trading_enabled() is False
