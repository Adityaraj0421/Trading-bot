"""Alerting — severity-aware alert funnel for the trading agent.

Routes alerts to Telegram (critical) or log-only (warn/info).
Attach a ``Notifier`` instance via the constructor for Telegram delivery.
Safe to use with no notifier (tests, paper mode).
"""
from __future__ import annotations

import logging
from typing import Any, Literal

_log = logging.getLogger(__name__)

AlertSeverity = Literal["info", "warn", "critical"]

_LIQUIDATION_PROXIMITY_THRESHOLD = 0.10  # 10% from liquidation price


class Alerting:
    """Severity-aware alert funnel.

    Args:
        notifier: Optional ``Notifier`` instance. When provided, ``critical``
            alerts are forwarded to ``notifier.notify_error()``. When absent,
            all alerts go to the Python logger only.
    """

    def __init__(self, notifier: Any = None) -> None:
        self._notifier = notifier

    def alert(self, msg: str, severity: AlertSeverity = "warn") -> None:
        """Fire an alert with the given severity.

        Args:
            msg: Human-readable alert message.
            severity: One of ``"info"``, ``"warn"``, ``"critical"``.
                ``"critical"`` also sends a Telegram notification if a
                notifier is wired.
        """
        if severity == "critical":
            _log.error("ALERT[CRITICAL]: %s", msg)
            if self._notifier is not None:
                self._notifier.notify_error("agent", f"CRITICAL: {msg}")
        elif severity == "warn":
            _log.warning("ALERT[WARN]: %s", msg)
        else:
            _log.info("ALERT[INFO]: %s", msg)

    def liquidation_proximity(
        self,
        symbol: str,
        mark_price: float,
        liquidation_price: float,
    ) -> None:
        """Fire a critical alert when mark price is within 10% of liquidation.

        Args:
            symbol: Trading pair, e.g. ``"BTC/USDT"``.
            mark_price: Current mark price.
            liquidation_price: Position liquidation price.
        """
        if liquidation_price <= 0:
            return
        proximity = abs(mark_price - liquidation_price) / liquidation_price
        if proximity < _LIQUIDATION_PROXIMITY_THRESHOLD:
            self.alert(
                f"{symbol} liquidation proximity {proximity:.1%} "
                f"(mark={mark_price:,.0f}, liq={liquidation_price:,.0f})",
                severity="critical",
            )
