"""
Interactive Telegram Bot for the Crypto Trading Agent
======================================================
Two-way Telegram integration:
  - Incoming: command handlers (/status, /balance, /positions, etc.)
  - Outgoing: trade confirmation inline keyboards (approve/reject)
  - Security: only responds to the configured TELEGRAM_CHAT_ID

Uses raw requests.post for sending (works from the sync agent thread).
Receives updates via FastAPI webhook endpoint (api/routes/telegram.py).
"""

import contextlib
import logging
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import requests

from config import Config

_log = logging.getLogger(__name__)

TELEGRAM_API = "https://api.telegram.org/bot{token}"


@dataclass
class PendingConfirmation:
    """A trade awaiting user approval via Telegram inline keyboard."""

    trade_id: str
    signal: str
    pair: str
    price: float
    quantity: float
    side: str
    strategy: str
    confidence: float
    created_at: datetime = field(default_factory=datetime.now)
    event: threading.Event = field(default_factory=threading.Event)
    decision: str | None = None  # "approved" / "rejected" / None
    message_id: int | None = None


class TelegramBot:
    """Interactive Telegram bot with commands and trade confirmations."""

    def __init__(self, data_store: Any = None) -> None:
        self._token: str = Config.TELEGRAM_BOT_TOKEN
        self._chat_id: str = str(Config.TELEGRAM_CHAT_ID)
        self._webhook_url: str = Config.TELEGRAM_WEBHOOK_URL
        self._webhook_secret: str = Config.TELEGRAM_WEBHOOK_SECRET
        self._data_store: Any = data_store

        # Pending trade confirmations: trade_id -> PendingConfirmation
        self._pending: dict[str, PendingConfirmation] = {}
        self._pending_lock = threading.Lock()

        # Command registry
        self._commands: dict[str, Callable] = {
            "start": self._cmd_help,
            "help": self._cmd_help,
            "status": self._cmd_status,
            "balance": self._cmd_balance,
            "positions": self._cmd_positions,
            "trades": self._cmd_trades,
            "pause": self._cmd_pause,
            "resume": self._cmd_resume,
            "close_all": self._cmd_close_all,
        }

        # Start cleanup thread for expired confirmations
        self._cleanup_thread = threading.Thread(target=self._cleanup_loop, daemon=True, name="tg-cleanup")
        self._cleanup_thread.start()

    @property
    def enabled(self) -> bool:
        """Whether the Telegram bot is configured with token and chat ID."""
        return bool(self._token and self._chat_id)

    def set_data_store(self, store: Any) -> None:
        """Attach the shared DataStore for command handlers."""
        self._data_store = store

    # ------------------------------------------------------------------
    # Webhook management
    # ------------------------------------------------------------------

    def setup_webhook(self) -> bool:
        """Register the webhook URL with Telegram."""
        if not self.enabled or not self._webhook_url:
            _log.info("Telegram bot: webhook not configured (no URL)")
            return False
        try:
            payload = {
                "url": self._webhook_url,
                "allowed_updates": ["message", "callback_query"],
            }
            if self._webhook_secret:
                payload["secret_token"] = self._webhook_secret
            resp = requests.post(
                f"{TELEGRAM_API.format(token=self._token)}/setWebhook",
                json=payload,
                timeout=10,
            )
            if resp.ok:
                _log.info("Telegram webhook registered: %s", self._webhook_url)
                return True
            _log.warning("Telegram setWebhook failed: %s", resp.text)
            return False
        except Exception as e:
            _log.warning("Telegram setWebhook error: %s", e)
            return False

    def teardown_webhook(self) -> None:
        """Remove the webhook from Telegram."""
        if not self.enabled:
            return
        try:
            requests.post(
                f"{TELEGRAM_API.format(token=self._token)}/deleteWebhook",
                timeout=10,
            )
            _log.info("Telegram webhook removed")
        except Exception as e:
            _log.debug("Telegram deleteWebhook error: %s", e)

    # ------------------------------------------------------------------
    # Incoming update handling (called by FastAPI webhook route)
    # ------------------------------------------------------------------

    def handle_update(self, update: dict) -> None:
        """Process an incoming Telegram update (message or callback query)."""
        try:
            if "callback_query" in update:
                self._handle_callback_query(update["callback_query"])
            elif "message" in update:
                self._handle_message(update["message"])
        except Exception as e:
            _log.warning("Telegram update handling error: %s", e, exc_info=True)

    def _handle_message(self, message: dict) -> None:
        """Route incoming text commands."""
        chat_id = str(message.get("chat", {}).get("id", ""))
        if chat_id != self._chat_id:
            _log.debug("Ignoring message from unauthorized chat: %s", chat_id)
            return

        text = message.get("text", "").strip()
        if not text.startswith("/"):
            return

        # Parse command: "/status" or "/status@botname"
        parts = text[1:].split()
        command = parts[0].split("@")[0].lower()

        handler = self._commands.get(command)
        if handler:
            handler()
        else:
            self._send("Unknown command. Use /help to see available commands.")

    def _handle_callback_query(self, callback: dict) -> None:
        """Process inline keyboard button presses (trade confirmations)."""
        chat_id = str(callback.get("message", {}).get("chat", {}).get("id", ""))
        if chat_id != self._chat_id:
            return

        callback_id = callback.get("id", "")
        data = callback.get("data", "")

        # Parse callback data: "approve:trade_id" or "reject:trade_id"
        # or "confirm_close_all" / "cancel_close_all"
        if data == "confirm_close_all":
            self._execute_close_all()
            self._answer_callback(callback_id, "Force close triggered")
            return
        elif data == "cancel_close_all":
            msg_id = callback.get("message", {}).get("message_id")
            if msg_id:
                self._edit_message(msg_id, "Close all cancelled.")
            self._answer_callback(callback_id, "Cancelled")
            return

        if ":" not in data:
            self._answer_callback(callback_id, "Invalid action")
            return

        action, trade_id = data.split(":", 1)
        with self._pending_lock:
            conf = self._pending.get(trade_id)

        if conf is None:
            self._answer_callback(callback_id, "Confirmation expired or already handled")
            return

        if action == "approve":
            conf.decision = "approved"
            conf.event.set()
            self._answer_callback(callback_id, "Trade APPROVED")
            if conf.message_id:
                self._edit_message(conf.message_id, f"APPROVED {conf.side.upper()} {conf.pair} @ ${conf.price:,.2f}")
        elif action == "reject":
            conf.decision = "rejected"
            conf.event.set()
            self._answer_callback(callback_id, "Trade REJECTED")
            if conf.message_id:
                self._edit_message(conf.message_id, f"REJECTED {conf.side.upper()} {conf.pair} @ ${conf.price:,.2f}")

    # ------------------------------------------------------------------
    # Command handlers
    # ------------------------------------------------------------------

    def _cmd_help(self) -> None:
        """Send the list of available bot commands."""
        msg = (
            "*Crypto Trading Agent*\n\n"
            "/status - Agent state, capital, PnL, regime\n"
            "/balance - Capital breakdown and exposure\n"
            "/positions - Open positions with stops\n"
            "/trades - Last 5 closed trades\n"
            "/pause - Halt all trading\n"
            "/resume - Resume trading\n"
            "/close\\_all - Force close all positions\n"
            "/help - Show this message"
        )
        self._send(msg, parse_mode="Markdown")

    def _cmd_status(self) -> None:
        """Send agent status summary (state, capital, PnL, regime)."""
        snap = self._get_snapshot()
        if not snap:
            self._send("Agent not running yet.")
            return
        snap = self._enrich_with_trade_db(snap)

        auto = snap.get("autonomous", {})
        state = auto.get("state", "unknown")
        state_emoji = {
            "normal": "🟢",
            "cautious": "🟡",
            "defensive": "🟠",
            "halted": "🔴",
        }.get(state, "⚪")

        pnl = snap.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        positions = snap.get("positions", [])

        msg = (
            f"{state_emoji} *Status: {state.upper()}*\n\n"
            f"Cycle: #{snap.get('cycle', 0)}\n"
            f"Capital: ${snap.get('capital', 0):,.2f}\n"
            f"PnL: {pnl_sign}${pnl:,.2f}\n"
            f"Win Rate: {(snap.get('win_rate', 0) * 100):.1f}%\n"
            f"Regime: {snap.get('regime', 'unknown')}\n"
            f"Open Positions: {len(positions)}\n"
            f"Mode: {snap.get('trading_mode', 'paper').upper()}"
        )
        self._send(msg, parse_mode="Markdown")

    def _cmd_balance(self) -> None:
        """Send capital breakdown and exposure."""
        snap = self._get_snapshot()
        if not snap:
            self._send("Agent not running yet.")
            return
        snap = self._enrich_with_trade_db(snap)

        capital = snap.get("capital", 0)
        pnl = snap.get("total_pnl", 0)
        pnl_sign = "+" if pnl >= 0 else ""
        portfolio = snap.get("portfolio", {})
        exposure = portfolio.get("total_exposure", 0)

        msg = (
            f"*Balance*\n\n"
            f"Capital: ${capital:,.2f}\n"
            f"Total PnL: {pnl_sign}${pnl:,.2f}\n"
            f"Exposure: ${exposure:,.2f}\n"
            f"Available: ${(capital - exposure):,.2f}"
        )

        # Per-pair exposure if available
        pair_exp = portfolio.get("pair_exposure", {})
        if pair_exp:
            msg += "\n\n*Exposure by Pair:*"
            for pair, exp in pair_exp.items():
                msg += f"\n  {pair}: ${exp:,.2f}"

        self._send(msg, parse_mode="Markdown")

    def _cmd_positions(self) -> None:
        """Send open positions with entry, stops, and unrealized PnL."""
        snap = self._get_snapshot()
        if not snap:
            self._send("Agent not running yet.")
            return

        positions = snap.get("positions", [])
        if not positions:
            self._send("No open positions.")
            return

        msg = f"*Open Positions ({len(positions)})*\n"
        for p in positions:
            side_emoji = "🟢" if p.get("side") == "long" else "🔴"
            unrealized = p.get("unrealized_pnl", 0)
            pnl_sign = "+" if unrealized >= 0 else ""
            msg += (
                f"\n{side_emoji} *{p.get('symbol', '?')}* {p.get('side', '?').upper()}\n"
                f"  Entry: ${p.get('entry_price', 0):,.2f}\n"
                f"  Qty: {p.get('quantity', 0):.6f}\n"
                f"  SL: ${p.get('stop_loss', 0):,.2f} | TP: ${p.get('take_profit', 0):,.2f}\n"
                f"  PnL: {pnl_sign}${unrealized:,.2f}\n"
                f"  Strategy: {p.get('strategy_name', '?')}"
            )

        self._send(msg, parse_mode="Markdown")

    def _cmd_trades(self) -> None:
        """Send the last 5 closed trades (persistent TradeDB preferred)."""
        if not self._data_store:
            self._send("Agent not running yet.")
            return

        # Prefer TradeDB (persistent) over in-memory log (session-only)
        trades: list[dict] = []
        db = self._data_store.get_trade_db()
        if db is not None:
            with contextlib.suppress(Exception):
                trades = db.get_trade_history(limit=5)
        if not trades:
            trades = self._data_store.get_trade_log()[-5:]

        if not trades:
            self._send("No closed trades yet.")
            return

        msg = f"*Recent Trades ({len(trades)})*\n"
        for t in trades:
            # TradeDB uses pnl_net/strategy_name; in-memory uses pnl/strategy
            pnl = t.get("pnl_net") or t.get("pnl", 0)
            pnl_emoji = "+" if pnl >= 0 else ""
            result = "WIN" if pnl >= 0 else "LOSS"
            strategy = t.get("strategy_name") or t.get("strategy", "?")
            msg += (
                f"\n{'✅' if pnl >= 0 else '❌'} *{t.get('symbol', '?')}* {t.get('side', '?').upper()}\n"
                f"  {result}: {pnl_emoji}${pnl:,.2f}\n"
                f"  Entry: ${t.get('entry_price', 0):,.2f} -> Exit: ${t.get('exit_price', 0):,.2f}\n"
                f"  Strategy: {strategy}"
            )

        self._send(msg, parse_mode="Markdown")

    def _cmd_pause(self) -> None:
        """Halt trading via the decision engine kill switch."""
        if not self._data_store:
            self._send("Agent not running yet.")
            return
        decision = self._data_store.get_decision_engine()
        if decision is None:
            self._send("Decision engine not available.")
            return
        decision.emergency_halt("Telegram /pause command")
        self._send("🔴 *Trading PAUSED*\nUse /resume to restart.", parse_mode="Markdown")

    def _cmd_resume(self) -> None:
        """Resume trading after a pause."""
        if not self._data_store:
            self._send("Agent not running yet.")
            return
        decision = self._data_store.get_decision_engine()
        if decision is None:
            self._send("Decision engine not available.")
            return
        decision.emergency_resume()
        self._send("🟢 *Trading RESUMED*", parse_mode="Markdown")

    def _cmd_close_all(self) -> None:
        """Send confirmation keyboard before force-closing all positions."""
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Yes, close all", "callback_data": "confirm_close_all"},
                    {"text": "Cancel", "callback_data": "cancel_close_all"},
                ]
            ]
        }
        self._send(
            "⚠️ *Force close ALL positions?*\nThis cannot be undone.",
            parse_mode="Markdown",
            reply_markup=keyboard,
        )

    def _execute_close_all(self) -> None:
        """Actually perform the force close after confirmation."""
        if not self._data_store:
            self._send("Agent not running.")
            return
        decision = self._data_store.get_decision_engine()
        if decision is None:
            self._send("Decision engine not available.")
            return
        decision.force_close_all_positions()
        self._send("🚨 *Force close triggered.* All positions will be closed on the next cycle.", parse_mode="Markdown")

    # ------------------------------------------------------------------
    # Trade confirmation flow (called from agent thread)
    # ------------------------------------------------------------------

    def request_confirmation(
        self, signal: str, pair: str, side: str, price: float, quantity: float, strategy: str, confidence: float
    ) -> PendingConfirmation:
        """
        Send a trade confirmation request to Telegram and return a
        PendingConfirmation. The caller should wait on conf.event with a timeout.
        """
        trade_id = uuid.uuid4().hex[:8]
        conf = PendingConfirmation(
            trade_id=trade_id,
            signal=signal,
            pair=pair,
            side=side,
            price=price,
            quantity=quantity,
            strategy=strategy,
            confidence=confidence,
        )

        with self._pending_lock:
            self._pending[trade_id] = conf

        side_emoji = "🟢" if side == "long" else "🔴"
        text = (
            f"{side_emoji} *Trade Confirmation*\n\n"
            f"Signal: {signal} {pair}\n"
            f"Side: {side.upper()}\n"
            f"Price: ${price:,.2f}\n"
            f"Quantity: {quantity:.6f}\n"
            f"Strategy: {strategy}\n"
            f"Confidence: {confidence:.0%}\n\n"
            f"_Timeout: {Config.TELEGRAM_CONFIRMATION_TIMEOUT}s "
            f"(default: {Config.TELEGRAM_CONFIRMATION_DEFAULT})_"
        )
        keyboard = {
            "inline_keyboard": [
                [
                    {"text": "Approve ✅", "callback_data": f"approve:{trade_id}"},
                    {"text": "Reject ❌", "callback_data": f"reject:{trade_id}"},
                ]
            ]
        }

        msg_id = self._send(text, parse_mode="Markdown", reply_markup=keyboard, return_message_id=True)
        conf.message_id = msg_id

        return conf

    # ------------------------------------------------------------------
    # Telegram API helpers (raw requests.post — works from sync threads)
    # ------------------------------------------------------------------

    def _send(
        self,
        text: str,
        parse_mode: str | None = None,
        reply_markup: dict | None = None,
        return_message_id: bool = False,
    ) -> int | None:
        """Send a message to the configured chat."""
        if not self.enabled:
            return None
        try:
            payload = {
                "chat_id": self._chat_id,
                "text": text,
                "disable_web_page_preview": True,
            }
            if parse_mode:
                payload["parse_mode"] = parse_mode
            if reply_markup:
                payload["reply_markup"] = reply_markup

            resp = requests.post(
                f"{TELEGRAM_API.format(token=self._token)}/sendMessage",
                json=payload,
                timeout=10,
            )
            if resp.ok and return_message_id:
                return resp.json().get("result", {}).get("message_id")
            if not resp.ok:
                _log.debug("Telegram send failed: %s", resp.text)
        except Exception as e:
            _log.debug("Telegram send error: %s", e)
        return None

    def _edit_message(self, message_id: int, text: str) -> None:
        """Edit an existing message (used to update confirmation status)."""
        if not self.enabled:
            return
        try:
            requests.post(
                f"{TELEGRAM_API.format(token=self._token)}/editMessageText",
                json={
                    "chat_id": self._chat_id,
                    "message_id": message_id,
                    "text": text,
                },
                timeout=10,
            )
        except Exception as e:
            _log.debug("Telegram edit error: %s", e)

    def _answer_callback(self, callback_id: str, text: str) -> None:
        """Acknowledge an inline button press."""
        if not self.enabled:
            return
        try:
            requests.post(
                f"{TELEGRAM_API.format(token=self._token)}/answerCallbackQuery",
                json={"callback_query_id": callback_id, "text": text},
                timeout=10,
            )
        except Exception as e:
            _log.debug("Telegram callback answer error: %s", e)

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_snapshot(self) -> dict[str, Any]:
        """Get agent snapshot from DataStore."""
        if not self._data_store:
            return {}
        return self._data_store.get_snapshot()

    def _enrich_with_trade_db(self, snap: dict[str, Any]) -> dict[str, Any]:
        """Merge persistent TradeDB stats into snapshot when session data is incomplete."""
        if not self._data_store:
            return snap
        db = self._data_store.get_trade_db()
        if db is None:
            return snap
        try:
            db_stats = db.get_total_stats()
            if db_stats and db_stats.get("total_trades", 0) > snap.get("total_trades", 0):
                snap = snap.copy()
                snap["total_pnl"] = round(db_stats.get("total_pnl") or 0, 2)
                snap["total_trades"] = db_stats["total_trades"]
                # DB returns win_rate as percentage; snapshot uses 0-1 fraction
                snap["win_rate"] = round((db_stats.get("win_rate") or 0) / 100, 4)
        except Exception:
            pass
        return snap

    def _cleanup_loop(self) -> None:
        """Remove expired pending confirmations every 30 seconds."""
        while True:
            time.sleep(30)
            try:
                timeout = Config.TELEGRAM_CONFIRMATION_TIMEOUT
                now = datetime.now()
                expired = []
                with self._pending_lock:
                    for tid, conf in self._pending.items():
                        age = (now - conf.created_at).total_seconds()
                        if age > timeout + 30:  # grace period
                            expired.append(tid)
                    for tid in expired:
                        del self._pending[tid]
                if expired:
                    _log.debug("Cleaned up %d expired confirmations", len(expired))
            except Exception as e:
                _log.warning("Confirmation cleanup error: %s", e, exc_info=True)
