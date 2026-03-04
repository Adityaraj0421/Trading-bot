"""Tests for alerting.py severity-aware alert funnel."""
from unittest.mock import MagicMock


class TestAlerting:
    def test_critical_calls_notifier_error(self):
        """critical severity triggers notifier.notify_error()."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.alert("disk full", severity="critical")
        notifier.notify_error.assert_called_once()
        call_msg = notifier.notify_error.call_args[0][1]
        assert "disk full" in call_msg

    def test_warn_does_not_call_notifier(self):
        """warn severity logs but does not call notifier."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.alert("high latency", severity="warn")
        notifier.notify_error.assert_not_called()

    def test_no_notifier_does_not_raise(self):
        """Alerting with no notifier is safe for all severities."""
        from alerting import Alerting
        al = Alerting()
        al.alert("test", severity="critical")  # should not raise
        al.alert("test", severity="warn")

    def test_liquidation_proximity_alert(self):
        """liquidation_proximity() fires critical when within threshold."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        # mark price 9% above liquidation → within 10% threshold
        al.liquidation_proximity("BTC/USDT", mark_price=54_450.0, liquidation_price=50_000.0)
        notifier.notify_error.assert_called_once()

    def test_liquidation_proximity_no_alert_when_safe(self):
        """liquidation_proximity() is silent when > 10% away."""
        from alerting import Alerting
        notifier = MagicMock()
        al = Alerting(notifier=notifier)
        al.liquidation_proximity("BTC/USDT", mark_price=60_000.0, liquidation_price=50_000.0)
        notifier.notify_error.assert_not_called()
