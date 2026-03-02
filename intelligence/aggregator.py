"""
Intelligence Aggregator — combines all signal sources into a single adjustment.
Produces an adjustment_factor (0.5 to 1.5) and bias (bullish/bearish/neutral).

This module orchestrates multi-source signal aggregation across all intelligence
providers (on-chain, order book, correlation, whale, news, LLM sentiment,
funding/OI, liquidation, and cascade prediction) and fuses them into a single
composite signal used by the decision engine.
"""

from __future__ import annotations

from typing import Any

from intelligence.cascade_predictor import CascadePredictor
from intelligence.correlation import CorrelationAnalyzer
from intelligence.fear_greed import FearGreedProvider
from intelligence.funding_oi import FundingOIAnalyzer
from intelligence.liquidation import LiquidationAnalyzer
from intelligence.llm_sentiment import LLMSentimentProvider
from intelligence.news_sentiment import NewsSentimentAnalyzer
from intelligence.onchain import OnChainAnalyzer
from intelligence.orderbook import OrderBookAnalyzer
from intelligence.whale_tracker import WhaleTracker


class IntelligenceAggregator:
    """Combines all intelligence signals into a single trading adjustment.

    Instantiates all sub-providers and exposes a single ``get_signals()``
    method that collects their outputs, computes a net bullish/bearish score,
    and maps it to an ``adjustment_factor`` (0.5–1.5) used by the decision
    engine to scale strategy confidence.
    """

    def __init__(self, exchange: Any = None) -> None:
        """Initialise all intelligence sub-providers.

        Args:
            exchange: An optional CCXT exchange instance passed to providers
                that require live order-book access (e.g. OrderBookAnalyzer).
                If ``None``, order-book analysis is skipped gracefully.
        """
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
            FearGreedProvider(),
        ]
        self._last_signals: list[dict[str, Any]] = []

    def get_signals(self) -> dict[str, Any]:
        """Collect signals from all providers and compute aggregate.

        Calls ``get_signal()`` on every registered provider, catches any
        exceptions (so a single failing provider cannot abort the whole
        aggregation), and computes a composite score.

        Returns:
            Dictionary with the following keys:

            - ``adjustment_factor`` (float): Mapped score in [0.5, 1.5].
              1.0 = neutral, >1.0 = bullish tilt, <1.0 = bearish tilt.
            - ``bias`` (str): ``'bullish'``, ``'bearish'``, or ``'neutral'``.
            - ``bullish_score`` (float): Sum of bullish signal strengths.
            - ``bearish_score`` (float): Sum of bearish signal strengths.
            - ``net_score`` (float): ``bullish_score - bearish_score``.
            - ``signals`` (list[dict]): Raw signal dicts from each provider,
              each containing ``source``, ``signal``, ``strength``, ``data``.
        """
        signals = []
        for provider in self.providers:
            try:
                sig = provider.get_signal()
                signals.append(sig)
            except Exception as e:
                signals.append(
                    {
                        "source": provider.__class__.__name__,
                        "signal": "neutral",
                        "strength": 0.0,
                        "data": {"error": str(e)},
                    }
                )

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
        """Return the last computed signals without triggering a new fetch.

        Returns:
            The list of provider signal dicts produced by the most recent
            call to ``get_signals()``.  Returns an empty list if
            ``get_signals()`` has not been called yet.
        """
        return self._last_signals
