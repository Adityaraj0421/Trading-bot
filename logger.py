"""
Structured Logging Module
==========================
Dual-output logging: structured JSON to file, pretty console output.
Tracks all signals, trades, regime changes, and model events.
"""

from __future__ import annotations

import json
import logging
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any

from config import Config


class StructuredLogger:
    """
    Production-grade logger that writes:
    1. Console: human-readable format
    2. File: JSON-lines for machine parsing / monitoring (rotated at 20MB, 5 backups)
    """

    def __init__(self, name: str = "trading_agent") -> None:
        """Initialise the structured logger with console and rotating file handlers.

        Args:
            name: Logger namespace prefix used for the console and file
                sub-loggers.  Defaults to ``"trading_agent"``.
        """
        self.name = name
        self._setup_console_logger()
        self._setup_file_logger()
        # v7.0: bounded in-memory event buffer (was unbounded list)
        self.events: deque[dict[str, Any]] = deque(maxlen=2000)

    def _setup_console_logger(self) -> None:
        """Configure the human-readable console handler."""
        self.console = logging.getLogger(f"{self.name}.console")
        self.console.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        if not self.console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"))
            self.console.addHandler(handler)

    def _setup_file_logger(self) -> None:
        """Configure the rotating JSON-lines file handler."""
        self.file_logger = logging.getLogger(f"{self.name}.file")
        self.file_logger.setLevel(logging.DEBUG)
        if not self.file_logger.handlers:
            # v7.0: Rotating log — 20 MB per file, keep 5 backups (100 MB total max)
            handler = RotatingFileHandler(
                Config.LOG_FILE,
                maxBytes=20 * 1024 * 1024,
                backupCount=5,
                encoding="utf-8",
            )
            handler.setFormatter(logging.Formatter("%(message)s"))
            self.file_logger.addHandler(handler)

    def _log_event(self, event_type: str, data: dict[str, Any], level: str = "INFO") -> dict[str, Any]:
        """Build a structured event, append it to the in-memory buffer, and write to file.

        Args:
            event_type: Short category string (e.g. ``"trade_open"``).
            data: Payload fields merged into the event dictionary.
            level: Log severity string — ``"INFO"``, ``"WARNING"``, or
                ``"ERROR"``.

        Returns:
            The complete event dictionary including ``timestamp``, ``type``,
            and ``level`` keys in addition to all *data* fields.
        """
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "level": level,
            **data,
        }
        self.events.append(event)
        self.file_logger.info(json.dumps(event))
        return event

    # --- Specific event loggers ---

    def log_cycle_start(self, cycle: int, price: float, pair: str) -> None:
        """Log the start of a new trading cycle.

        Args:
            cycle: Monotonically increasing cycle counter.
            price: Latest close price for *pair*.
            pair: Trading pair symbol, e.g. ``"BTC/USDT"``.
        """
        self._log_event(
            "cycle_start",
            {
                "cycle": cycle,
                "price": price,
                "pair": pair,
            },
        )
        self.console.info(f"Cycle #{cycle} | {pair} = ${price:,.2f}")

    def log_signal(self, signal: str, confidence: float, source: str, regime: str = "", strategy: str = "") -> None:
        """Log a trading signal with its source and confidence.

        Args:
            signal: Signal direction — ``"BUY"``, ``"SELL"``, or ``"HOLD"``.
            confidence: Signal confidence in the range [0, 1].
            source: Component that generated the signal (e.g. strategy name).
            regime: Market regime label at signal generation time.
            strategy: Strategy identifier when *source* is a composite label.
        """
        self._log_event(
            "signal",
            {
                "signal": signal,
                "confidence": confidence,
                "source": source,
                "regime": regime,
                "strategy": strategy,
            },
        )

    def log_trade_open(
        self,
        symbol: str,
        side: str,
        price: float,
        quantity: float,
        sl: float,
        tp: float,
        trailing: float,
        strategy: str = "",
    ) -> None:
        """Log a new trade entry with position details and risk levels.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Trade direction — ``"buy"`` or ``"sell"``.
            price: Entry execution price.
            quantity: Asset quantity traded.
            sl: Stop-loss price level.
            tp: Take-profit price level.
            trailing: Initial trailing-stop price level.
            strategy: Name of the strategy that generated the signal.
        """
        self._log_event(
            "trade_open",
            {
                "symbol": symbol,
                "side": side,
                "price": price,
                "quantity": quantity,
                "stop_loss": sl,
                "take_profit": tp,
                "trailing_stop": trailing,
                "strategy": strategy,
            },
        )
        self.console.info(
            f"OPEN {side.upper()} {symbol} | qty={quantity:.6f} @ ${price:,.2f} | "
            f"SL=${sl:,.2f} TP=${tp:,.2f} Trail=${trailing:,.2f}"
        )

    def log_trade_close(
        self,
        symbol: str,
        side: str,
        entry: float,
        exit_price: float,
        pnl_net: float,
        pnl_gross: float,
        fees: float,
        reason: str,
        hold_bars: int,
        strategy: str = "",
    ) -> None:
        """Log a trade exit with PnL, fees, and exit reason.

        Args:
            symbol: Trading pair symbol, e.g. ``"BTC/USDT"``.
            side: Trade direction — ``"buy"`` or ``"sell"``.
            entry: Entry execution price.
            exit_price: Exit execution price.
            pnl_net: Net profit/loss after fees and slippage.
            pnl_gross: Gross profit/loss before fees and slippage.
            fees: Total fees paid on the round trip.
            reason: Human-readable exit reason (e.g. ``"stop_loss"``).
            hold_bars: Number of OHLCV bars the position was held.
            strategy: Name of the strategy that generated the entry signal.
        """
        level = "INFO" if pnl_net >= 0 else "WARNING"
        self._log_event(
            "trade_close",
            {
                "symbol": symbol,
                "side": side,
                "entry_price": entry,
                "exit_price": exit_price,
                "pnl_net": round(pnl_net, 4),
                "pnl_gross": round(pnl_gross, 4),
                "fees": round(fees, 4),
                "reason": reason,
                "hold_bars": hold_bars,
                "strategy": strategy,
            },
            level=level,
        )
        icon = "+" if pnl_net >= 0 else "-"
        self.console.info(
            f"[{icon}] CLOSE {side.upper()} {symbol} | "
            f"PnL=${pnl_net:,.2f} (gross ${pnl_gross:,.2f}) | "
            f"{reason} | {hold_bars} bars"
        )

    def log_regime_change(self, old_regime: str, new_regime: str, confidence: float) -> None:
        """Log a market regime transition.

        Args:
            old_regime: Previous regime label (e.g. ``"trending_up"``).
            new_regime: Newly detected regime label.
            confidence: Detector confidence in the new regime, in [0, 1].
        """
        self._log_event(
            "regime_change",
            {
                "old_regime": old_regime,
                "new_regime": new_regime,
                "confidence": confidence,
            },
        )
        self.console.info(f"Regime: {old_regime} -> {new_regime} ({confidence:.0%})")

    def log_model_train(self, accuracy: float, samples: int, drift_detected: bool = False) -> None:
        """Log an ML model training event with accuracy and drift status.

        Args:
            accuracy: Validation accuracy of the freshly trained model,
                in the range [0, 1].
            samples: Number of training samples used.
            drift_detected: When True, the log level is elevated to WARNING
                and a console warning is emitted.
        """
        level = "WARNING" if drift_detected else "INFO"
        self._log_event(
            "model_train",
            {
                "accuracy": accuracy,
                "samples": samples,
                "drift_detected": drift_detected,
            },
            level=level,
        )
        if drift_detected:
            self.console.warning(f"Model drift detected! Retraining (acc: {accuracy:.2%})")
        else:
            self.console.info(f"Model trained: accuracy={accuracy:.2%}, samples={samples}")

    def log_error(self, component: str, error: str) -> None:
        """Log an error from a named component.

        Args:
            component: Name of the subsystem that raised the error
                (e.g. ``"DataFetcher"``).
            error: Human-readable error message or exception string.
        """
        self._log_event(
            "error",
            {
                "component": component,
                "error": error,
            },
            level="ERROR",
        )
        self.console.error(f"[{component}] {error}")

    def log_portfolio(self, capital: float, positions: int, total_pnl: float, fees: float, win_rate: float) -> None:
        """Log a portfolio snapshot with capital, positions, and PnL.

        Args:
            capital: Available cash balance.
            positions: Number of currently open positions.
            total_pnl: Cumulative realised net PnL.
            fees: Cumulative fees paid across all closed trades.
            win_rate: Fraction of winning trades, in [0, 1].
        """
        self._log_event(
            "portfolio",
            {
                "capital": round(capital, 2),
                "open_positions": positions,
                "total_pnl": round(total_pnl, 2),
                "total_fees": round(fees, 2),
                "win_rate": round(win_rate, 4),
            },
        )

    def get_recent_events(self, event_type: str | None = None, limit: int = 50) -> list[dict[str, Any]]:
        """Return recent events from the in-memory buffer, optionally filtered by type.

        Args:
            event_type: When provided, only events whose ``type`` field
                matches this value are returned.
            limit: Maximum number of events to return (from the tail of the
                buffer).

        Returns:
            List of event dictionaries ordered from oldest to newest within
            the returned slice.
        """
        events = list(self.events)
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]
