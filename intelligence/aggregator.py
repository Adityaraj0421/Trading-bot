"""
Intelligence Aggregator — combines all signal sources into a single adjustment.
Produces an adjustment_factor (0.5 to 1.5) and bias (bullish/bearish/neutral).
"""

from typing import Any

from intelligence.onchain import OnChainAnalyzer
from intelligence.orderbook import OrderBookAnalyzer
from intelligence.correlation import CorrelationAnalyzer
from intelligence.whale_tracker import WhaleTracker
from intelligence.news_sentiment import NewsSentimentAnalyzer
from intelligence.llm_sentiment import LLMSentimentProvider
from intelligence.funding_oi import FundingOIAnalyzer
from intelligence.liquidation import LiquidationAnalyzer
from intelligence.cascade_predictor import CascadePredictor


class IntelligenceAggregator:
    """Combines all intelligence signals into a single trading adjustment."""

    def __init__(self, exchange: Any = None) -> None:
        from config import Config
        # Convert trading pairs to Binance format for derivatives providers
        binance_symbols = [p.replace("/", "") for p in Config.TRADING_PAIRS]

        self.providers = [
            OnChainAnalyzer(),
            OrderBookAnalyzer(exchange=exchange),
            CorrelationAnalyzer(),
            WhaleTracker(),
            NewsSentimentAnalyzer(),
            LLMSentimentProvider(),
            FundingOIAnalyzer(symbols=binance_symbols),
            LiquidationAnalyzer(symbols=binance_symbols),
            CascadePredictor(symbols=binance_symbols),
        ]
        self._last_signals = []

    def get_signals(self) -> dict[str, Any]:
        """Collect signals from all providers and compute aggregate."""
        signals = []
        for provider in self.providers:
            try:
                sig = provider.get_signal()
                signals.append(sig)
            except Exception as e:
                signals.append({
                    "source": provider.__class__.__name__,
                    "signal": "neutral",
                    "strength": 0.0,
                    "data": {"error": str(e)},
                })

        self._last_signals = signals

        # Compute aggregate
        bullish_score = 0.0
        bearish_score = 0.0
        for sig in signals:
            if sig["signal"] == "bullish":
                bullish_score += sig["strength"]
            elif sig["signal"] == "bearish":
                bearish_score += sig["strength"]

        net_score = bullish_score - bearish_score  # -1 to +1 range typically

        # Map to adjustment_factor: 0.5 (very bearish) to 1.5 (very bullish)
        adjustment_factor = 1.0 + max(-0.5, min(0.5, net_score))

        if net_score > 0.1:
            bias = "bullish"
        elif net_score < -0.1:
            bias = "bearish"
        else:
            bias = "neutral"

        return {
            "adjustment_factor": round(adjustment_factor, 3),
            "bias": bias,
            "bullish_score": round(bullish_score, 3),
            "bearish_score": round(bearish_score, 3),
            "net_score": round(net_score, 3),
            "signals": signals,
        }

    def get_last_signals(self) -> list[dict[str, Any]]:
        """Return the last computed signals."""
        return self._last_signals
