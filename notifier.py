"""
Notification & Alerting System
================================
Sends alerts via Telegram, Discord, and email for:
  - Trade executions (open/close)
  - System state changes (NORMAL -> CAUTIOUS -> DEFENSIVE -> HALTED)
  - Large losses or winning streaks
  - Kill switch activations
  - Daily performance summaries

All channels are optional — configure via .env.
"""

import contextlib
import logging
import smtplib
import threading
from collections import deque
from datetime import datetime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from typing import Any

import requests

from config import Config

_log = logging.getLogger(__name__)


class AlertLevel:
    """Constants for notification severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    TRADE = "trade"
    DAILY_SUMMARY = "daily_summary"


class Notifier:
    """Multi-channel notification system for the trading agent."""

    def __init__(self) -> None:
        self._telegram_token: str = getattr(Config, "TELEGRAM_BOT_TOKEN", "")
        self._telegram_chat_id: str = getattr(Config, "TELEGRAM_CHAT_ID", "")
        self._discord_webhook: str = getattr(Config, "DISCORD_WEBHOOK_URL", "")
        self._email_enabled: bool = getattr(Config, "EMAIL_ALERTS_ENABLED", False)
        self._email_smtp: str = getattr(Config, "EMAIL_SMTP_HOST", "")
        self._email_port: int = int(getattr(Config, "EMAIL_SMTP_PORT", 587))
        self._email_user: str = getattr(Config, "EMAIL_USER", "")
        self._email_pass: str = getattr(Config, "EMAIL_PASS", "")
        self._email_to: str = getattr(Config, "EMAIL_TO", "")

        self._history: deque[dict] = deque(maxlen=500)
        self._daily_trades: list[dict] = []
        self._daily_pnl: float = 0.0

        # Rate limiting
        self._last_send: dict[str, float] = {}  # channel -> timestamp
        self._min_interval: int = 2  # Min seconds between messages per channel

        # Optional DataStore for /notifications API
        self._data_store: Any = None

    def set_data_store(self, store: Any) -> None:
        """Attach DataStore so sent notifications appear on the /notifications API."""
        self._data_store = store

    @property
    def telegram_enabled(self) -> bool:
        """Whether Telegram notifications are configured."""
        return bool(self._telegram_token and self._telegram_chat_id)

    @property
    def discord_enabled(self) -> bool:
        """Whether Discord webhook notifications are configured."""
        return bool(self._discord_webhook)

    @property
    def email_enabled(self) -> bool:
        """Whether email SMTP notifications are configured."""
        return bool(self._email_enabled and self._email_smtp and self._email_user)

    def has_channels(self) -> bool:
        """Check if any notification channel is configured."""
        return self.telegram_enabled or self.discord_enabled or self.email_enabled

    def notify_trade_open(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        sl: float = 0,
        tp: float = 0,
        strategy: str = "",
        confidence: float = 0,
    ) -> None:
        """Notify about a new trade."""
        side_emoji = "🟢" if side == "long" else "🔴"
        msg = (
            f"{side_emoji} **TRADE OPEN** | {side.upper()} {symbol}\n"
            f"Price: ${price:,.2f} | Qty: {quantity:.6f}\n"
            f"SL: ${sl:,.2f} | TP: ${tp:,.2f}\n"
            f"Strategy: {strategy} | Conf: {confidence:.0%}"
        )
        self._send_all(msg, AlertLevel.TRADE)

    def notify_trade_close(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float,
        pnl: float,
        reason: str,
        strategy: str,
        hold_bars: int,
    ) -> None:
        """Notify about a closed trade."""
        pnl_emoji = "✅" if pnl >= 0 else "❌"
        pnl_sign = "+" if pnl >= 0 else ""
        msg = (
            f"{pnl_emoji} **TRADE CLOSED** | {side.upper()} {symbol}\n"
            f"Entry: ${entry:,.2f} -> Exit: ${exit_price:,.2f}\n"
            f"PnL: {pnl_sign}${pnl:,.2f} | Reason: {reason}\n"
            f"Strategy: {strategy} | Duration: {hold_bars} bars"
        )
        self._send_all(msg, AlertLevel.TRADE)

        # Track for daily summary
        self._daily_trades.append(
            {
                "symbol": symbol,
                "side": side,
                "pnl": pnl,
                "strategy": strategy,
                "reason": reason,
            }
        )
        self._daily_pnl += pnl

    def notify_state_change(self, old_state: str, new_state: str, reason: str) -> None:
        """Notify about system state change."""
        state_emojis = {
            "normal": "🟢",
            "cautious": "🟡",
            "defensive": "🟠",
            "halted": "🔴",
        }
        emoji = state_emojis.get(new_state, "⚪")
        msg = f"{emoji} **STATE CHANGE**: {old_state.upper()} → {new_state.upper()}\nReason: {reason}"
        level = AlertLevel.CRITICAL if new_state in ("halted", "defensive") else AlertLevel.WARNING
        self._send_all(msg, level)

    def notify_kill_switch(self, reason: str) -> None:
        """Notify about kill switch activation."""
        msg = f"🚨 **EMERGENCY HALT** 🚨\nAll trading stopped.\nReason: {reason}"
        self._send_all(msg, AlertLevel.CRITICAL)

    def notify_large_loss(self, pnl: float, capital_pct: float) -> None:
        """Notify about a large loss."""
        msg = f"⚠️ **LARGE LOSS**: ${pnl:,.2f} ({capital_pct:.1f}% of capital)\nReview positions immediately."
        self._send_all(msg, AlertLevel.WARNING)

    def notify_daily_summary(self, capital: float, total_pnl: float, win_rate: float, open_positions: int) -> None:
        """Send daily performance summary."""
        pnl_emoji = "📈" if total_pnl >= 0 else "📉"
        trade_count = len(self._daily_trades)
        winners = sum(1 for t in self._daily_trades if t["pnl"] > 0)

        strategy_pnl = {}
        for t in self._daily_trades:
            s = t["strategy"]
            strategy_pnl[s] = strategy_pnl.get(s, 0) + t["pnl"]

        strat_lines = (
            "\n".join(
                f"  {s}: {'+' if p >= 0 else ''}${p:,.2f}" for s, p in sorted(strategy_pnl.items(), key=lambda x: -x[1])
            )
            if strategy_pnl
            else "  No trades today"
        )

        pnl_sign = "+" if self._daily_pnl >= 0 else ""
        msg = (
            f"{pnl_emoji} **DAILY SUMMARY** | {datetime.now().strftime('%Y-%m-%d')}\n"
            f"Capital: ${capital:,.2f}\n"
            f"Daily PnL: {pnl_sign}${self._daily_pnl:,.2f}\n"
            f"Trades: {trade_count} ({winners} wins)\n"
            f"Open Positions: {open_positions}\n"
            f"Overall Win Rate: {win_rate:.1f}%\n\n"
            f"**By Strategy:**\n{strat_lines}"
        )
        self._send_all(msg, AlertLevel.DAILY_SUMMARY)

        # Reset daily tracking
        self._daily_trades = []
        self._daily_pnl = 0.0

    def notify_heartbeat(
        self,
        cycle: int,
        capital: float,
        total_pnl: float,
        open_positions: int,
        pairs: list[str],
        prices: dict[str, float] | None = None,
    ) -> None:
        """v7.0: Hourly heartbeat — agent is alive + quick stats."""
        pnl_sign = "+" if total_pnl >= 0 else ""
        daily_sign = "+" if self._daily_pnl >= 0 else ""

        price_lines = ""
        if prices:
            price_lines = "\n".join(f"  {p.split('/')[0]}: ${v:,.2f}" for p, v in prices.items())
            price_lines = f"\n**Prices:**\n{price_lines}\n"

        msg = (
            f"💓 **HEARTBEAT** | Cycle #{cycle}\n"
            f"Capital: ${capital:,.2f} | PnL: {pnl_sign}${total_pnl:,.2f}\n"
            f"Session PnL: {daily_sign}${self._daily_pnl:,.2f} ({len(self._daily_trades)} trades)\n"
            f"Open: {open_positions} position{'s' if open_positions != 1 else ''}"
            f"{price_lines}"
        )
        self._send_all(msg, AlertLevel.INFO)

    def notify_error(self, component: str, error: str) -> None:
        """Notify about a system error."""
        msg = f"⚙️ **ERROR** in {component}:\n{error[:200]}"
        self._send_all(msg, AlertLevel.WARNING)

    def _send_all(self, message: str, level: str) -> None:
        """Send to all enabled channels (async, non-blocking)."""
        ts = datetime.now().isoformat()
        self._history.append(
            {
                "ts": ts,
                "level": level,
                "message": message[:200],
            }
        )

        # Push to DataStore for the /notifications API endpoint
        if self._data_store is not None:
            with contextlib.suppress(Exception):
                self._data_store.append_notification({
                    "timestamp": ts,
                    "level": level,
                    "message": message[:200],
                })

        # Send in background threads to avoid blocking
        if self.telegram_enabled:
            threading.Thread(
                target=self._send_telegram,
                args=(message,),
                daemon=True,
            ).start()

        if self.discord_enabled:
            threading.Thread(
                target=self._send_discord,
                args=(message,),
                daemon=True,
            ).start()

        if self.email_enabled and level in (AlertLevel.CRITICAL, AlertLevel.DAILY_SUMMARY):
            threading.Thread(
                target=self._send_email,
                args=(message, level),
                daemon=True,
            ).start()

    def _rate_limited(self, channel: str) -> bool:
        """Check if we're sending too fast."""
        now = datetime.now().timestamp()
        last = self._last_send.get(channel, 0)
        if now - last < self._min_interval:
            return True
        self._last_send[channel] = now
        return False

    def _send_telegram(self, message: str) -> None:
        """Send via Telegram Bot API."""
        if self._rate_limited("telegram"):
            return
        try:
            # Convert markdown bold to Telegram format
            text = message.replace("**", "*")
            resp = requests.post(
                f"https://api.telegram.org/bot{self._telegram_token}/sendMessage",
                json={
                    "chat_id": self._telegram_chat_id,
                    "text": text,
                    "parse_mode": "Markdown",
                    "disable_web_page_preview": True,
                },
                timeout=10,
            )
            if not resp.ok:
                _log.debug("Telegram send failed: %s", resp.text)
        except Exception as e:
            _log.debug("Telegram error: %s", e)

    def _send_discord(self, message: str) -> None:
        """Send via Discord webhook."""
        if self._rate_limited("discord"):
            return
        try:
            resp = requests.post(
                self._discord_webhook,
                json={"content": message},
                timeout=10,
            )
            if not resp.ok:
                _log.debug("Discord send failed: %s", resp.text)
        except Exception as e:
            _log.debug("Discord error: %s", e)

    def _send_email(self, message: str, level: str) -> None:
        """Send via email (SMTP)."""
        if self._rate_limited("email"):
            return
        try:
            msg = MIMEMultipart()
            msg["From"] = self._email_user
            msg["To"] = self._email_to
            subject_prefix = "🚨 ALERT" if level == AlertLevel.CRITICAL else "📊 Summary"
            msg["Subject"] = f"[Trading Bot] {subject_prefix} - {datetime.now().strftime('%Y-%m-%d %H:%M')}"

            # Convert to HTML
            html_body = message.replace("\n", "<br>").replace("**", "<b>", 1)
            while "**" in html_body:
                html_body = html_body.replace("**", "</b>", 1).replace("**", "<b>", 1)

            msg.attach(MIMEText(html_body, "html"))

            with smtplib.SMTP(self._email_smtp, self._email_port) as server:
                server.starttls()
                server.login(self._email_user, self._email_pass)
                server.send_message(msg)

        except Exception as e:
            _log.debug("Email error: %s", e)

    def get_history(self, limit: int = 50) -> list[dict]:
        """Get recent notification history."""
        return list(self._history)[-limit:]
