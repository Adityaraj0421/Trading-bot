"""Tests for the arbitrage module."""
import pytest
from unittest.mock import MagicMock, patch
from arbitrage.fee_calculator import FeeCalculator
from arbitrage.opportunity_detector import ArbitrageDetector
from arbitrage.price_monitor import PriceMonitor
from arbitrage.latency_tracker import LatencyTracker
from arbitrage.execution_engine import ArbitrageExecutor


class TestFeeCalculator:
    def test_binance_trading_fee(self):
        calc = FeeCalculator()
        fee = calc.trading_fee("binance", 1000.0, "taker")
        assert fee == 1.0  # 0.1%

    def test_unknown_exchange_default(self):
        calc = FeeCalculator()
        fee = calc.trading_fee("unknown_exchange", 1000.0)
        assert fee == 2.0  # 0.2% default

    def test_round_trip_cost(self):
        calc = FeeCalculator()
        cost = calc.total_round_trip_cost("binance", "kraken", 1000.0)
        assert cost > 0

    def test_fee_summary(self):
        calc = FeeCalculator()
        summary = calc.get_fee_summary()
        assert "trading_fees" in summary
        assert "withdrawal_fees" in summary
        assert "binance" in summary["trading_fees"]


class TestPriceMonitor:
    def test_empty_exchanges(self):
        monitor = PriceMonitor({})
        prices = monitor.fetch_all_prices("BTC/USDT")
        assert prices == {}

    def test_get_last_prices_empty(self):
        monitor = PriceMonitor({})
        assert monitor.get_last_prices() == {}


class TestArbitrageDetector:
    def test_no_exchanges(self):
        detector = ArbitrageDetector({})
        opps = detector.scan("BTC/USDT")
        assert opps == []
        assert detector.get_status()["scan_count"] == 1

    def test_status_structure(self):
        detector = ArbitrageDetector({})
        status = detector.get_status()
        assert "scan_count" in status
        assert "active_opportunities" in status
        assert "exchanges_monitored" in status
        assert "fees" in status


class TestLatencyTracker:
    def test_record_and_stats(self):
        tracker = LatencyTracker()
        tracker.record("binance", 50.0)
        tracker.record("binance", 75.0)
        stats = tracker.get_stats("binance")
        assert stats["samples"] == 2
        assert stats["avg_ms"] == 62.5

    def test_empty_stats(self):
        tracker = LatencyTracker()
        stats = tracker.get_stats("unknown")
        assert stats["samples"] == 0

    def test_max_samples(self):
        tracker = LatencyTracker(max_samples=5)
        for i in range(10):
            tracker.record("binance", float(i))
        stats = tracker.get_stats("binance")
        assert stats["samples"] == 5

    def test_get_all_stats(self):
        tracker = LatencyTracker()
        tracker.record("binance", 50.0)
        tracker.record("kraken", 100.0)
        all_stats = tracker.get_all_stats()
        assert len(all_stats) == 2


class TestArbitrageExecutor:
    def test_paper_execute(self):
        executor = ArbitrageExecutor()
        executor.set_capital(10000.0)
        result = executor.execute({
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "pair": "BTC/USDT",
            "buy_price": 95000.0,
            "sell_price": 95500.0,
            "spread_pct": 0.005,
            "net_profit_pct": 0.002,
        })
        assert result["status"] == "paper_executed"
        assert result["mode"] == "paper"
        assert result["net_pnl"] is not None
        assert result["fees_paid"] > 0
        assert result["quantity"] > 0
        assert len(executor.get_execution_log()) == 1

    def test_no_capital_skipped(self):
        executor = ArbitrageExecutor()
        result = executor.execute({
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "pair": "BTC/USDT",
            "buy_price": 95000.0,
            "sell_price": 95500.0,
        })
        assert result["status"] == "skipped"

    def test_summary_tracking(self):
        executor = ArbitrageExecutor()
        executor.set_capital(10000.0)
        executor.execute({
            "buy_exchange": "binance",
            "sell_exchange": "kraken",
            "pair": "BTC/USDT",
            "buy_price": 95000.0,
            "sell_price": 95500.0,
            "spread_pct": 0.005,
            "net_profit_pct": 0.002,
        })
        summary = executor.get_summary()
        assert summary["total_trades"] == 1
        assert summary["total_fees"] > 0
        assert summary["total_pnl"] != 0
