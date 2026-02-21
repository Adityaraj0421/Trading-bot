"""
Arbitrage Execution Engine — simultaneous buy/sell on different exchanges.
Paper mode simulates capital allocation, slippage, and PnL tracking.
"""

from datetime import datetime
from arbitrage.fee_calculator import FeeCalculator
from arbitrage.latency_tracker import LatencyTracker
from config import Config


class ArbitrageExecutor:
    """Executes arbitrage trades with paper simulation and PnL tracking."""

    def __init__(self, exchanges: dict = None):
        self.exchanges = exchanges or {}
        self.fee_calculator = FeeCalculator()
        self.latency_tracker = LatencyTracker()
        self.execution_log: list[dict] = []
        self.capital = 0.0
        self.total_pnl = 0.0
        self.total_fees = 0.0
        self.successful_trades = 0
        self.failed_trades = 0
        self._max_concurrent = 1
        self._active_trades = 0

    def set_capital(self, capital: float):
        """Set available capital for arbitrage trading."""
        self.capital = capital

    def execute(self, opportunity: dict) -> dict:
        """
        Execute an arbitrage opportunity.
        Paper mode: simulates order placement with slippage and fee deduction.
        """
        if self._active_trades >= self._max_concurrent:
            return {"status": "skipped", "reason": "max concurrent trades reached"}

        buy_exchange = opportunity.get("buy_exchange", "")
        sell_exchange = opportunity.get("sell_exchange", "")
        buy_price = opportunity.get("buy_price", 0)
        sell_price = opportunity.get("sell_price", 0)
        net_profit_pct = opportunity.get("net_profit_pct", 0)

        if buy_price <= 0 or sell_price <= 0:
            return {"status": "failed", "reason": "invalid prices"}

        # Position size: use a fraction of capital
        trade_capital = min(self.capital * 0.1, self.capital)
        if trade_capital <= 0:
            return {"status": "skipped", "reason": "insufficient capital"}

        quantity = trade_capital / buy_price

        # Simulate slippage (0.05% on each side)
        slippage = Config.SLIPPAGE_PCT
        effective_buy = buy_price * (1 + slippage)
        effective_sell = sell_price * (1 - slippage)

        # Calculate fees
        buy_fee = self.fee_calculator.trading_fee(buy_exchange, trade_capital, "taker")
        sell_fee = self.fee_calculator.trading_fee(sell_exchange, quantity * effective_sell, "taker")
        total_fees = buy_fee + sell_fee

        # Gross and net PnL
        gross_pnl = (effective_sell - effective_buy) * quantity
        net_pnl = gross_pnl - total_fees

        self._active_trades += 1

        # Simulate execution delay (paper mode completes instantly)
        result = {
            "status": "paper_executed",
            "buy_exchange": buy_exchange,
            "sell_exchange": sell_exchange,
            "pair": opportunity.get("pair", Config.TRADING_PAIR),
            "buy_price": round(effective_buy, 2),
            "sell_price": round(effective_sell, 2),
            "quantity": round(quantity, 8),
            "trade_capital": round(trade_capital, 2),
            "spread_pct": opportunity.get("spread_pct", 0),
            "net_profit_pct": round(net_pnl / trade_capital * 100, 4) if trade_capital > 0 else 0,
            "gross_pnl": round(gross_pnl, 4),
            "net_pnl": round(net_pnl, 4),
            "fees_paid": round(total_fees, 4),
            "slippage_applied": slippage,
            "executed_at": datetime.now().isoformat(),
            "mode": "paper",
        }

        # Update tracking
        self.total_pnl += net_pnl
        self.total_fees += total_fees
        self.capital += net_pnl
        if net_pnl > 0:
            self.successful_trades += 1
        else:
            self.failed_trades += 1

        self._active_trades -= 1
        self.execution_log.append(result)
        return result

    def get_execution_log(self) -> list[dict]:
        return self.execution_log.copy()

    def get_summary(self) -> dict:
        total_trades = self.successful_trades + self.failed_trades
        return {
            "total_trades": total_trades,
            "successful": self.successful_trades,
            "failed": self.failed_trades,
            "win_rate": self.successful_trades / total_trades if total_trades > 0 else 0,
            "total_pnl": round(self.total_pnl, 4),
            "total_fees": round(self.total_fees, 4),
            "capital": round(self.capital, 2),
        }
