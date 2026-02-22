"""
Trade execution module.
Handles paper trading simulation and live order placement via CCXT.
"""

import logging
from datetime import datetime
from typing import Any

import ccxt

from config import Config

_log = logging.getLogger(__name__)


class PaperExecutor:
    """Simulates trade execution without touching real money."""

    def __init__(self) -> None:
        self.orders: list[dict[str, Any]] = []

    def place_order(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        """Simulate placing an order. Fills instantly at current price."""
        order = {
            "id": f"paper_{len(self.orders) + 1}",
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
        """Cancel a paper order (always succeeds)."""
        return True


class LiveExecutor:
    """Places real orders on the exchange via CCXT."""

    def __init__(self, exchange: ccxt.Exchange) -> None:
        self.exchange = exchange

    def place_order(self, symbol: str, side: str, quantity: float, price: float) -> dict[str, Any]:
        """Place a limit order on the exchange."""
        if side not in ("long", "short"):
            raise ValueError(f"Invalid side '{side}': must be 'long' or 'short'")
        try:
            if side == "long":
                order = self.exchange.create_limit_buy_order(symbol, quantity, price)
            else:
                order = self.exchange.create_limit_sell_order(symbol, quantity, price)

            _log.info(
                "[Live] %s %.6f %s @ $%,.2f (order: %s)",
                side.upper(),
                quantity,
                symbol,
                price,
                order["id"],
            )
            return order
        except ccxt.InsufficientFunds as e:
            _log.error("[Live] Insufficient funds: %s", e)
            return {"error": str(e)}
        except ccxt.ExchangeError as e:
            _log.error("[Live] Exchange error: %s", e)
            return {"error": str(e)}

    def cancel_order(self, order_id: str, symbol: str | None = None) -> bool:
        """Cancel an open order on the exchange.

        Returns:
            True if the order was cancelled, False on failure.
        """
        effective_symbol = symbol or Config.TRADING_PAIR
        if symbol is None:
            _log.warning(
                "[Live] cancel_order called without symbol — falling back to primary pair %s",
                effective_symbol,
            )
        try:
            self.exchange.cancel_order(order_id, effective_symbol)
            return True
        except Exception as e:
            _log.error("[Live] Cancel error: %s", e)
            return False
