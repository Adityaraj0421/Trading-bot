"""
Tests for Notifier — multi-channel notification system.
All external sends mocked (requests.post, smtplib).
"""

import time
import pytest
from unittest.mock import patch, MagicMock
from notifier import Notifier, AlertLevel


@pytest.fixture()
def notifier_disabled():
    """Notifier with no channels enabled."""
    return Notifier()


@pytest.fixture()
def notifier_telegram(monkeypatch):
    """Notifier with only Telegram enabled."""
    monkeypatch.setattr("config.Config.TELEGRAM_BOT_TOKEN", "test-token")
    monkeypatch.setattr("config.Config.TELEGRAM_CHAT_ID", "12345")
    return Notifier()


@pytest.fixture()
def notifier_discord(monkeypatch):
    """Notifier with only Discord enabled."""
    monkeypatch.setattr("config.Config.DISCORD_WEBHOOK_URL", "https://discord.com/api/webhooks/test")
    return Notifier()


# ── Channel Detection ─────────────────────────────────────────────


class TestChannelDetection:
    def test_no_channels_by_default(self, notifier_disabled):
        assert notifier_disabled.has_channels() is False
        assert notifier_disabled.telegram_enabled is False
        assert notifier_disabled.discord_enabled is False
        assert notifier_disabled.email_enabled is False

    def test_telegram_enabled(self, notifier_telegram):
        assert notifier_telegram.telegram_enabled is True
        assert notifier_telegram.has_channels() is True

    def test_discord_enabled(self, notifier_discord):
        assert notifier_discord.discord_enabled is True
        assert notifier_discord.has_channels() is True


# ── Notification Methods ──────────────────────────────────────────


class TestNotificationMethods:
    @patch("notifier.requests.post")
    def test_notify_trade_open_sends_telegram(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        notifier_telegram.notify_trade_open(
            symbol="BTC/USDT", side="long", price=50000,
            quantity=0.1, sl=49000, tp=52000,
            strategy="momentum", confidence=0.85,
        )
        # Give daemon thread a moment
        time.sleep(0.1)
        assert mock_post.called

    @patch("notifier.requests.post")
    def test_notify_trade_close_tracks_daily(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        notifier_telegram.notify_trade_close(
            symbol="BTC/USDT", side="long", entry=50000,
            exit_price=51000, pnl=100, reason="take_profit",
            strategy="momentum", hold_bars=10,
        )
        time.sleep(0.1)
        assert len(notifier_telegram._daily_trades) == 1
        assert notifier_telegram._daily_pnl == 100

    @patch("notifier.requests.post")
    def test_notify_state_change(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        notifier_telegram.notify_state_change("normal", "cautious", "3 losses")
        time.sleep(0.1)
        assert mock_post.called

    @patch("notifier.requests.post")
    def test_notify_kill_switch(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        notifier_telegram.notify_kill_switch("manual halt")
        time.sleep(0.1)
        assert mock_post.called


# ── History ───────────────────────────────────────────────────────


class TestHistory:
    def test_notification_stored_in_history(self, notifier_disabled):
        notifier_disabled.notify_error("test_component", "test error")
        history = notifier_disabled.get_history()
        assert len(history) == 1
        assert history[0]["level"] == AlertLevel.WARNING

    def test_history_limit(self, notifier_disabled):
        for i in range(5):
            notifier_disabled.notify_error("comp", f"error {i}")
        assert len(notifier_disabled.get_history(limit=3)) == 3


# ── Rate Limiting ─────────────────────────────────────────────────


class TestRateLimiting:
    @patch("notifier.requests.post")
    def test_rapid_sends_rate_limited(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        notifier_telegram._min_interval = 5  # 5 second minimum

        notifier_telegram._send_telegram("first message")
        notifier_telegram._send_telegram("second message — should be rate limited")

        # Only the first should go through
        assert mock_post.call_count == 1


# ── Daily Summary ─────────────────────────────────────────────────


class TestDailySummary:
    @patch("notifier.requests.post")
    def test_daily_summary_resets_tracking(self, mock_post, notifier_telegram):
        mock_post.return_value = MagicMock(ok=True)
        # Add some trades
        notifier_telegram._daily_trades = [
            {"symbol": "BTC/USDT", "side": "long", "pnl": 100, "strategy": "mom", "reason": "tp"},
            {"symbol": "BTC/USDT", "side": "short", "pnl": -30, "strategy": "rev", "reason": "sl"},
        ]
        notifier_telegram._daily_pnl = 70

        notifier_telegram.notify_daily_summary(
            capital=10000, total_pnl=500, win_rate=65.0, open_positions=1,
        )

        # Daily tracking should be reset
        assert notifier_telegram._daily_trades == []
        assert notifier_telegram._daily_pnl == 0.0


# ── Discord ───────────────────────────────────────────────────────


class TestDiscord:
    @patch("notifier.requests.post")
    def test_discord_send(self, mock_post, notifier_discord):
        mock_post.return_value = MagicMock(ok=True)
        notifier_discord.notify_error("test", "boom")
        time.sleep(0.1)
        assert mock_post.called
        # Verify Discord payload format
        call_args = mock_post.call_args
        assert "content" in call_args.kwargs.get("json", call_args[1].get("json", {}))
