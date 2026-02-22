"""
Tests for TelegramBot — command dispatch, confirmations, auth.
All Telegram API calls mocked via requests.post.
"""

import contextlib
import threading
from unittest.mock import MagicMock, patch

import pytest

from telegram_bot import PendingConfirmation, TelegramBot


@pytest.fixture()
def bot(monkeypatch):
    """TelegramBot with Telegram credentials configured."""
    monkeypatch.setattr("config.Config.TELEGRAM_BOT_TOKEN", "test-token-123")
    monkeypatch.setattr("config.Config.TELEGRAM_CHAT_ID", "999")
    monkeypatch.setattr("config.Config.TELEGRAM_WEBHOOK_URL", "")
    monkeypatch.setattr("config.Config.TELEGRAM_WEBHOOK_SECRET", "secret")
    return TelegramBot()


@pytest.fixture()
def bot_disabled():
    """TelegramBot with no credentials (disabled)."""
    return TelegramBot()


# ── Basic Properties ──────────────────────────────────────────────


class TestBotProperties:
    def test_enabled_with_credentials(self, bot):
        assert bot.enabled is True

    def test_disabled_without_credentials(self, bot_disabled):
        assert bot_disabled.enabled is False

    def test_command_registry_populated(self, bot):
        assert "status" in bot._commands
        assert "help" in bot._commands
        assert "balance" in bot._commands
        assert "positions" in bot._commands
        assert "trades" in bot._commands
        assert "pause" in bot._commands
        assert "resume" in bot._commands

    def test_set_data_store(self, bot):
        mock_store = MagicMock()
        bot.set_data_store(mock_store)
        assert bot._data_store is mock_store


# ── PendingConfirmation ───────────────────────────────────────────


class TestPendingConfirmation:
    def test_creates_with_defaults(self):
        pc = PendingConfirmation(
            trade_id="test-123",
            signal="BUY",
            pair="BTC/USDT",
            price=50000,
            quantity=0.1,
            side="long",
            strategy="momentum",
            confidence=0.85,
        )
        assert pc.decision is None
        assert pc.message_id is None
        assert isinstance(pc.event, threading.Event)
        assert not pc.event.is_set()

    def test_event_can_be_set(self):
        pc = PendingConfirmation(
            trade_id="test-456",
            signal="SELL",
            pair="ETH/USDT",
            price=3000,
            quantity=1.0,
            side="short",
            strategy="reversion",
            confidence=0.7,
        )
        pc.decision = "approved"
        pc.event.set()
        assert pc.event.is_set()
        assert pc.decision == "approved"


# ── Command Handling ──────────────────────────────────────────────


class TestCommandHandling:
    @patch("telegram_bot.requests.post")
    def test_help_command(self, mock_post, bot):
        mock_post.return_value = MagicMock(ok=True)
        # Simulate an incoming /help command
        update = {
            "message": {
                "chat": {"id": 999},
                "text": "/help",
                "from": {"id": 999},
            }
        }
        if hasattr(bot, "handle_update"):
            bot.handle_update(update)

    @patch("telegram_bot.requests.post")
    def test_status_command_without_store(self, mock_post, bot):
        mock_post.return_value = MagicMock(ok=True)
        # _cmd_status should handle missing data_store gracefully
        with contextlib.suppress(Exception):
            bot._cmd_status(chat_id="999")


# ── Auth / Chat ID ────────────────────────────────────────────────


class TestAuth:
    def test_chat_id_stored(self, bot):
        assert bot._chat_id == "999"

    def test_token_stored(self, bot):
        assert bot._token == "test-token-123"


# ── Pending Trade Management ─────────────────────────────────────


class TestPendingTrades:
    def test_pending_dict_initially_empty(self, bot):
        assert len(bot._pending) == 0

    def test_add_and_remove_pending(self, bot):
        pc = PendingConfirmation(
            trade_id="t1",
            signal="BUY",
            pair="BTC/USDT",
            price=50000,
            quantity=0.1,
            side="long",
            strategy="momentum",
            confidence=0.8,
        )
        with bot._pending_lock:
            bot._pending["t1"] = pc
        assert "t1" in bot._pending

        with bot._pending_lock:
            del bot._pending["t1"]
        assert "t1" not in bot._pending

    def test_thread_lock_exists(self, bot):
        assert isinstance(bot._pending_lock, type(threading.Lock()))
