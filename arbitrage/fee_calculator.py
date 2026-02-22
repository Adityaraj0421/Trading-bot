"""
Fee Calculator — per-exchange trading and withdrawal fees.
Hardcoded defaults for Binance, Coinbase, Kraken.
"""


class FeeCalculator:
    """Calculates trading and withdrawal fees per exchange."""

    # Default fee schedules (maker/taker as proportion)
    TRADING_FEES = {
        "binance": {"maker": 0.001, "taker": 0.001},
        "coinbase": {"maker": 0.004, "taker": 0.006},
        "kraken": {"maker": 0.0016, "taker": 0.0026},
    }

    # Approximate BTC withdrawal fees
    WITHDRAWAL_FEES = {
        "binance": 0.0001,  # BTC
        "coinbase": 0.0,  # Free for Coinbase Pro
        "kraken": 0.00015,  # BTC
    }

    def trading_fee(self, exchange: str, amount: float, side: str = "taker") -> float:
        """Calculate trading fee for an exchange."""
        fees = self.TRADING_FEES.get(exchange, {"maker": 0.002, "taker": 0.002})
        rate = fees.get(side, fees["taker"])
        return amount * rate

    def withdrawal_fee(self, exchange: str) -> float:
        """Get withdrawal fee in BTC for an exchange."""
        return self.WITHDRAWAL_FEES.get(exchange, 0.0005)

    def total_round_trip_cost(self, buy_exchange: str, sell_exchange: str, amount: float) -> float:
        """Calculate total cost of buy on one exchange + sell on another."""
        buy_fee = self.trading_fee(buy_exchange, amount, "taker")
        sell_fee = self.trading_fee(sell_exchange, amount, "taker")
        transfer_fee = self.withdrawal_fee(buy_exchange)
        return buy_fee + sell_fee + transfer_fee

    def get_fee_summary(self) -> dict:
        """Return all fee schedules."""
        return {
            "trading_fees": self.TRADING_FEES,
            "withdrawal_fees": self.WITHDRAWAL_FEES,
        }
