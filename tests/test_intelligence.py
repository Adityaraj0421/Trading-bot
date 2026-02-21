"""Tests for the intelligence module.

v9.0: Aggregator expanded from 5 to 9 providers (added LLM, FundingOI,
Liquidation, Cascade). New providers don't have ENABLE_* flags — they
use network calls and catch exceptions. Aggregator tests now mock all
provider get_signal() calls to test aggregation logic in isolation.
"""
import pytest
from unittest.mock import patch, MagicMock
from intelligence.aggregator import IntelligenceAggregator
from intelligence.onchain import OnChainAnalyzer
from intelligence.orderbook import OrderBookAnalyzer
from intelligence.news_sentiment import NewsSentimentAnalyzer
from intelligence.whale_tracker import WhaleTracker
from intelligence.correlation import CorrelationAnalyzer


class TestOnChainAnalyzer:
    def test_disabled_returns_neutral(self):
        with patch("intelligence.onchain.Config") as mock_cfg:
            mock_cfg.ENABLE_ONCHAIN = False
            analyzer = OnChainAnalyzer()
            result = analyzer.get_signal()
            assert result["signal"] == "neutral"
            assert result["strength"] == 0.0


class TestOrderBookAnalyzer:
    def test_disabled_returns_neutral(self):
        with patch("intelligence.orderbook.Config") as mock_cfg:
            mock_cfg.ENABLE_ORDERBOOK = False
            analyzer = OrderBookAnalyzer()
            result = analyzer.get_signal()
            assert result["signal"] == "neutral"

    def test_no_exchange_returns_neutral(self):
        with patch("intelligence.orderbook.Config") as mock_cfg:
            mock_cfg.ENABLE_ORDERBOOK = True
            analyzer = OrderBookAnalyzer(exchange=None)
            result = analyzer.get_signal()
            assert result["signal"] == "neutral"


class TestNewsSentiment:
    def test_disabled_returns_neutral(self):
        with patch("intelligence.news_sentiment.Config") as mock_cfg:
            mock_cfg.ENABLE_NEWS_NLP = False
            analyzer = NewsSentimentAnalyzer()
            result = analyzer.get_signal()
            assert result["signal"] == "neutral"


class TestWhaleTracker:
    def test_disabled_returns_neutral(self):
        with patch("intelligence.whale_tracker.Config") as mock_cfg:
            mock_cfg.ENABLE_WHALE_TRACKING = False
            tracker = WhaleTracker()
            result = tracker.get_signal()
            assert result["signal"] == "neutral"


class TestCorrelationAnalyzer:
    def test_disabled_returns_neutral(self):
        with patch("intelligence.correlation.Config") as mock_cfg:
            mock_cfg.ENABLE_CORRELATION = False
            analyzer = CorrelationAnalyzer()
            result = analyzer.get_signal()
            assert result["signal"] == "neutral"


def _make_neutral_signal(source_name="test"):
    return {"source": source_name, "signal": "neutral", "strength": 0.0}


class TestAggregator:
    """v9.0: Aggregator now has 9 providers (was 5).

    New providers (LLM, FundingOI, Liquidation, Cascade) don't have
    ENABLE_* flags — they make network calls and catch exceptions.
    We mock all provider get_signal() calls to test aggregation logic.
    """

    def _create_mocked_aggregator(self, signals=None):
        """Create an aggregator with all providers mocked to return neutral."""
        with patch("config.Config") as agg_cfg:
            agg_cfg.TRADING_PAIRS = ["BTC/USDT"]
            agg = IntelligenceAggregator()

        if signals is None:
            signals = [_make_neutral_signal() for _ in range(9)]
        for i, provider in enumerate(agg.providers):
            provider.get_signal = MagicMock(
                return_value=signals[i] if i < len(signals) else _make_neutral_signal()
            )
        return agg

    def test_all_disabled_returns_neutral(self):
        agg = self._create_mocked_aggregator()
        result = agg.get_signals()
        assert result["adjustment_factor"] == 1.0
        assert result["bias"] == "neutral"
        # v9.0: 9 providers (was 5)
        assert len(result["signals"]) == 9

    def test_adjustment_factor_range(self):
        """Adjustment factor should always be between 0.5 and 1.5."""
        agg = self._create_mocked_aggregator()
        result = agg.get_signals()
        assert 0.5 <= result["adjustment_factor"] <= 1.5

    def test_signals_structure(self):
        agg = self._create_mocked_aggregator()
        result = agg.get_signals()
        for sig in result["signals"]:
            assert "source" in sig
            assert "signal" in sig
            assert "strength" in sig
            assert sig["signal"] in ("bullish", "bearish", "neutral")

    def test_bullish_signals_raise_factor(self):
        signals = [
            {"source": "test", "signal": "bullish", "strength": 0.8}
        ] + [_make_neutral_signal() for _ in range(8)]
        agg = self._create_mocked_aggregator(signals=signals)
        result = agg.get_signals()
        assert result["adjustment_factor"] > 1.0
        assert result["bias"] == "bullish"

    def test_bearish_signals_lower_factor(self):
        signals = [
            {"source": "test", "signal": "bearish", "strength": 0.8}
        ] + [_make_neutral_signal() for _ in range(8)]
        agg = self._create_mocked_aggregator(signals=signals)
        result = agg.get_signals()
        assert result["adjustment_factor"] < 1.0
        assert result["bias"] == "bearish"
