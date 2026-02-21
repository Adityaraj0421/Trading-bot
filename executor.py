"""
Trade execution module.
Handles paper trading simulation and live order placement via CCXT.
"""

import ccxt
from datetime import datetime
from config import Config
from risk_manager import RiskManager, Position


class PaperExecutor:
    """Simulates trade execution without touching real money."""

    def __init__(self):
        self.orders = []

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
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
        print(f"[Paper] 📝 {side.upper()} {quantity:.6f} {symbol} @ ${price:,.2f}")
        return order

    def cancel_order(self, order_id: str) -> bool:
        return True


class LiveExecutor:
    """Places real orders on the exchange via CCXT."""

    def __init__(self, exchange: ccxt.Exchange):
        self.exchange = exchange

    def place_order(
        self, symbol: str, side: str, quantity: float, price: float
    ) -> dict:
        """Place a limit order on the exchange."""
        try:
            if side == "long":
                order = self.exchange.create_limit_buy_order(symbol, quantity, price)
            else:
                order = self.exchange.create_limit_sell_order(symbol, quantity, price)

            print(
                f"[Live] 🔴 {side.upper()} {quantity:.6f} {symbol} @ ${price:,.2f} "
                f"(order: {order['id']})"
            )
            return order
        except ccxt.InsufficientFunds as e:
            print(f"[Live] Insufficient funds: {e}")
            return {"error": str(e)}
        except ccxt.ExchangeError as e:
            print(f"[Live] Exchange error: {e}")
            return {"error": str(e)}

    def cancel_order(self, order_id: str, symbol: str = None) -> bool:
        try:
            self.exchange.cancel_order(order_id, symbol or Config.TRADING_PAIR)
            return True
        except Exception as e:
            print(f"[Live] Cancel error: {e}")
            return False
