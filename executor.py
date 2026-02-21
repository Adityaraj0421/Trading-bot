"""
Trade execution module.
Handles paper trading simulation and live order placement via CCXT.
"""

import logging
from typing import Any

import ccxt
from datetime import datetime
from config import Config
from risk_manager import RiskManager, Position

_log = logging.getLogger(__name__)


class PaperExecutor:
    """Simulates trade execution without touching real money."""

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Simulate placing an order. Fills instantly at current price."""
        order = {
            "id": f"paper_{len(self.orders)+1}",
            "symbol": symbol,
            "side": side,
            "quantity": quantity,
            "price": price,
            "status": "filled",
            "timestamp": datetime.now().isoformat(),
            "mode": "PAPER",
        }
        self.orders.append(order)
        _log.info("[Paper] %s %.6f %s @ $%,.2f", side.upper(), quantity, symbol, price)
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True


class LiveExecutor:
    """Places real orders on the exchange via CCXT."""

    def __init__(self, exchange: ccxt.Exchange) -> None:
        self.exchange = exchange

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict[str, Any]:
        """Place a limit order on the exchange."""
        try:
            if side == "long":
                order = self.exchange.create_limit_buy_order(symbol, quantity, price)
            else:
                order = self.exchange.create_limit_sell_order(symbol, quantity, price)

            _log.info(
                "[Live] %s %.6f %s @ $%,.2f (order: %s)",
                side.upper(), quantity, symbol, price, order['id'],
            )
            return order
        except ccxt.InsufficientFunds as e:
            _log.error("[Live] Insufficient funds: %s", e)
            return {"error": str(e)}
        except ccxt.ExchangeError as e:
            _log.error("[Live] Exchange error: %s", e)
            return {"error": str(e)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol or Config.TRADING_PAIR)
            return True
        except Exception as e:
            _log.error("[Live] Cancel error: %s", e)
            return False
