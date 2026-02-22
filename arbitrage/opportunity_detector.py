"""
Arbitrage Opportunity Detector — finds profitable spread differences.
"""

from dataclasses import dataclass
from datetime import datetime

from arbitrage.fee_calculator import FeeCalculator
from arbitrage.price_monitor import PriceMonitor
from config import Config


@dataclass
class ArbitrageOpportunity:
    buy_exchange: str
    sell_exchange: str
    pair: str
    buy_price: float
    sell_price: float
    spread_pct: float
    net_profit_pct: float
    fees_pct: float
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "buy_exchange": self.buy_exchange,
            "sell_exchange": self.sell_exchange,
            "pair": self.pair,
            "buy_price": self.buy_price,
            "sell_price": self.sell_price,
            "spread_pct": round(self.spread_pct * 100, 4),
            "net_profit_pct": round(self.net_profit_pct * 100, 4),
            "fees_pct": round(self.fees_pct * 100, 4),
            "timestamp": self.timestamp,
        }


class ArbitrageDetector:
    """Detects arbitrage opportunities across exchanges."""

    def __init__(self, exchanges: dict = None):
        self.price_monitor = PriceMonitor(exchanges or {})
        self.fee_calculator = FeeCalculator()
        self._opportunities: list[ArbitrageOpportunity] = []
        self._scan_count = 0

    def scan(self, pair: str = None) -> list[ArbitrageOpportunity]:
        """Scan all exchange pairs for arbitrage opportunities."""
        pair = pair or Config.TRADING_PAIR
        prices = self.price_monitor.fetch_all_prices(pair)
        self._scan_count += 1

        opportunities = []
        exchange_names = [name for name, data in prices.items() if "error" not in data]

        for buy_ex in exchange_names:
            for sell_ex in exchange_names:
                if buy_ex == sell_ex:
                    continue

                buy_ask = prices[buy_ex].get("ask", 0)
                sell_bid = prices[sell_ex].get("bid", 0)

                if buy_ask <= 0 or sell_bid <= 0:
                    continue

                spread_pct = (sell_bid - buy_ask) / buy_ask

                # Calculate fees
                fees_pct = self.fee_calculator.TRADING_FEES.get(buy_ex, {}).get(
                    "taker", 0.002
                ) + self.fee_calculator.TRADING_FEES.get(sell_ex, {}).get("taker", 0.002)

                net_profit_pct = spread_pct - fees_pct

                if net_profit_pct > Config.ARBITRAGE_MIN_SPREAD_PCT:
                    opp = ArbitrageOpportunity(
                        buy_exchange=buy_ex,
                        sell_exchange=sell_ex,
                        pair=pair,
                        buy_price=buy_ask,
                        sell_price=sell_bid,
                        spread_pct=spread_pct,
                        net_profit_pct=net_profit_pct,
                        fees_pct=fees_pct,
                        timestamp=datetime.now().isoformat(),
                    )
                    opportunities.append(opp)

        self._opportunities = opportunities
        return opportunities

    def get_last_opportunities(self) -> list[dict]:
        return [o.to_dict() for o in self._opportunities]

    def get_status(self) -> dict:
        return {
            "scan_count": self._scan_count,
            "active_opportunities": len(self._opportunities),
            "exchanges_monitored": list(self.price_monitor.exchanges.keys()),
            "min_spread_pct": Config.ARBITRAGE_MIN_SPREAD_PCT * 100,
            "opportunities": self.get_last_opportunities(),
            "fees": self.fee_calculator.get_fee_summary(),
        }
