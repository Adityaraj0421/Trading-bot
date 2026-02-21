"""
Structured Logging Module
==========================
Dual-output logging: structured JSON to file, pretty console output.
Tracks all signals, trades, regime changes, and model events.
"""

import json
import logging
import os
from collections import deque
from datetime import datetime
from logging.handlers import RotatingFileHandler
from config import Config


class StructuredLogger:
    """
    Production-grade logger that writes:
    1. Console: human-readable format
    2. File: JSON-lines for machine parsing / monitoring (rotated at 20MB, 5 backups)
    """

    def __init__(self, name: str = "trading_agent"):
        self.name = name
        self._setup_console_logger()
        self._setup_file_logger()
        # v7.0: bounded in-memory event buffer (was unbounded list)
        self.events: deque = deque(maxlen=2000)

    def _setup_console_logger(self):
        self.console = logging.getLogger(f"{self.name}.console")
        self.console.setLevel(getattr(logging, Config.LOG_LEVEL, logging.INFO))
        if not self.console.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                "%(asctime)s [%(levelname)s] %(message)s", "%H:%M:%S"
            ))
            self.console.addHandler(handler)

    def _setup_file_logger(self):
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

    def _log_event(self, event_type: str, data: dict, level: str = "INFO"):
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

    def log_cycle_start(self, cycle: int, price: float, pair: str):
        self._log_event("cycle_start", {
            "cycle": cycle, "price": price, "pair": pair,
        })
        self.console.info(f"Cycle #{cycle} | {pair} = ${price:,.2f}")

    def log_signal(self, signal: str, confidence: float, source: str,
                   regime: str = "", strategy: str = ""):
        self._log_event("signal", {
            "signal": signal, "confidence": confidence,
            "source": source, "regime": regime, "strategy": strategy,
        })

    def log_trade_open(self, symbol: str, side: str, price: float,
                       quantity: float, sl: float, tp: float,
                       trailing: float, strategy: str = ""):
        self._log_event("trade_open", {
            "symbol": symbol, "side": side, "price": price,
            "quantity": quantity, "stop_loss": sl, "take_profit": tp,
            "trailing_stop": trailing, "strategy": strategy,
        })
        self.console.info(
            f"OPEN {side.upper()} {symbol} | qty={quantity:.6f} @ ${price:,.2f} | "
            f"SL=${sl:,.2f} TP=${tp:,.2f} Trail=${trailing:,.2f}"
        )

    def log_trade_close(self, symbol: str, side: str, entry: float,
                        exit_price: float, pnl_net: float, pnl_gross: float,
                        fees: float, reason: str, hold_bars: int,
                        strategy: str = ""):
        level = "INFO" if pnl_net >= 0 else "WARNING"
        self._log_event("trade_close", {
            "symbol": symbol, "side": side,
            "entry_price": entry, "exit_price": exit_price,
            "pnl_net": round(pnl_net, 4), "pnl_gross": round(pnl_gross, 4),
            "fees": round(fees, 4), "reason": reason,
            "hold_bars": hold_bars, "strategy": strategy,
        }, level=level)
        icon = "+" if pnl_net >= 0 else "-"
        self.console.info(
            f"[{icon}] CLOSE {side.upper()} {symbol} | "
            f"PnL=${pnl_net:,.2f} (gross ${pnl_gross:,.2f}) | "
            f"{reason} | {hold_bars} bars"
        )

    def log_regime_change(self, old_regime: str, new_regime: str,
                          confidence: float):
        self._log_event("regime_change", {
            "old_regime": old_regime, "new_regime": new_regime,
            "confidence": confidence,
        })
        self.console.info(f"Regime: {old_regime} -> {new_regime} ({confidence:.0%})")

    def log_model_train(self, accuracy: float, samples: int,
                        drift_detected: bool = False):
        level = "WARNING" if drift_detected else "INFO"
        self._log_event("model_train", {
            "accuracy": accuracy, "samples": samples,
            "drift_detected": drift_detected,
        }, level=level)
        if drift_detected:
            self.console.warning(f"Model drift detected! Retraining (acc: {accuracy:.2%})")
        else:
            self.console.info(f"Model trained: accuracy={accuracy:.2%}, samples={samples}")

    def log_error(self, component: str, error: str):
        self._log_event("error", {
            "component": component, "error": error,
        }, level="ERROR")
        self.console.error(f"[{component}] {error}")

    def log_portfolio(self, capital: float, positions: int, total_pnl: float,
                      fees: float, win_rate: float):
        self._log_event("portfolio", {
            "capital": round(capital, 2),
            "open_positions": positions,
            "total_pnl": round(total_pnl, 2),
            "total_fees": round(fees, 2),
            "win_rate": round(win_rate, 4),
        })

    def get_recent_events(self, event_type: str = None, limit: int = 50) -> list:
        """Query recent events, optionally filtered by type."""
        events = list(self.events)
        if event_type:
            events = [e for e in events if e["type"] == event_type]
        return events[-limit:]
