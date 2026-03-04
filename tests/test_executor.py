"""
Unit tests for PaperExecutor and LiveExecutor — trade execution (v2.0).
Covers order placement, cancellation, error handling, and edge cases.
"""

from datetime import datetime
from typing import Any
from unittest.mock import MagicMock, patch

import ccxt
import pytest

from executor import LiveExecutor, PaperExecutor


@pytest.fixture()
def executor():
    return PaperExecutor()


class TestPaperExecutor:
    """PaperExecutor: instant fill simulation."""

    def test_place_order_returns_filled(self, executor):
        order = executor.place_order("BTC/USDT", "long", 0.01, 50000.0)
        assert order["status"] == "filled"
        assert order["symbol"] == "BTC/USDT"
        assert order["side"] == "long"
        assert order["quantity"] == 0.01
        assert order["price"] == 50000.0
        assert order["mode"] == "PAPER"

    def test_order_ids_are_sequential(self, executor):
        o1 = executor.place_order("BTC/USDT", "long", 0.01, 50000.0)
        o2 = executor.place_order("ETH/USDT", "short", 1.0, 3000.0)
        o3 = executor.place_order("BTC/USDT", "long", 0.05, 51000.0)
        assert o1["id"] == "paper_1"
        assert o2["id"] == "paper_2"
        assert o3["id"] == "paper_3"

    def test_orders_accumulate(self, executor):
        assert len(executor.orders) == 0
        executor.place_order("BTC/USDT", "long", 0.01, 50000.0)
        executor.place_order("ETH/USDT", "short", 1.0, 3000.0)
        assert len(executor.orders) == 2

    def test_order_has_timestamp(self, executor):
        order = executor.place_order("BTC/USDT", "long", 0.01, 50000.0)
        assert "timestamp" in order
        assert len(order["timestamp"]) > 0
        # Verify timestamp is valid ISO format
        datetime.fromisoformat(order["timestamp"])

    def test_cancel_always_returns_true(self, executor):
        assert executor.cancel_order("paper_1") is True
        assert executor.cancel_order("nonexistent") is True

    def test_short_order(self, executor):
        order = executor.place_order("BTC/USDT", "short", 0.05, 48000.0)
        assert order["side"] == "short"
        assert order["price"] == 48000.0

    def test_multiple_symbols(self, executor):
        """Should handle orders for different trading pairs."""
        symbols = ["BTC/USDT", "ETH/USDT", "SOL/USDT", "XRP/USDT"]
        for i, sym in enumerate(symbols):
            result = executor.place_order(sym, "long", 0.1, 100.0 * (i + 1))
            assert result["symbol"] == sym
        assert len(executor.orders) == 4

    def test_high_volume_orders(self, executor):
        """Paper executor should handle many orders without issues."""
        for i in range(500):
            result = executor.place_order("BTC/USDT", "long", 0.001, 50000.0 + i)
            assert result["status"] == "filled"
        assert len(executor.orders) == 500

    def test_order_preservation_order(self, executor):
        """Orders should be stored in insertion order."""
        prices = [50000.0, 51000.0, 49000.0]
        for p in prices:
            executor.place_order("BTC/USDT", "long", 0.1, p)
        stored_prices = [o["price"] for o in executor.orders]
        assert stored_prices == prices

    def test_zero_quantity_order(self, executor):
        """Zero quantity order should still be recorded (validation is external)."""
        result = executor.place_order("BTC/USDT", "long", 0.0, 50000.0)
        assert result["status"] == "filled"
        assert result["quantity"] == 0.0


class TestLiveExecutor:
    """LiveExecutor: real exchange orders via CCXT mocks."""

    def setup_method(self):
        self.mock_exchange = MagicMock(spec=ccxt.Exchange)
        self.executor = LiveExecutor(self.mock_exchange)

    def test_place_long_order(self):
        """Long order should call create_limit_buy_order."""
        self.mock_exchange.create_limit_buy_order.return_value = {
            "id": "live_123",
            "status": "open",
        }
        result = self.executor.place_order("BTC/USDT", "long", 0.1, 50000.0)
        self.mock_exchange.create_limit_buy_order.assert_called_once_with("BTC/USDT", 0.1, 50000.0)
        assert result["id"] == "live_123"

    def test_place_short_order(self):
        """Short order should call create_limit_sell_order."""
        self.mock_exchange.create_limit_sell_order.return_value = {
            "id": "live_456",
            "status": "open",
        }
        result = self.executor.place_order("BTC/USDT", "short", 0.1, 50000.0)
        self.mock_exchange.create_limit_sell_order.assert_called_once_with("BTC/USDT", 0.1, 50000.0)
        assert result["id"] == "live_456"

    def test_insufficient_funds_returns_error(self):
        """InsufficientFunds exception should be caught and return error dict."""
        self.mock_exchange.create_limit_buy_order.side_effect = ccxt.InsufficientFunds("Not enough USDT")
        result = self.executor.place_order("BTC/USDT", "long", 100.0, 50000.0)
        assert "error" in result
        assert "Not enough USDT" in result["error"]

    def test_exchange_error_returns_error(self):
        """ExchangeError should be caught and return error dict."""
        self.mock_exchange.create_limit_sell_order.side_effect = ccxt.ExchangeError("Market closed")
        result = self.executor.place_order("BTC/USDT", "short", 0.1, 50000.0)
        assert "error" in result
        assert "Market closed" in result["error"]

    def test_cancel_order_success(self):
        """Successful cancel should return True."""
        self.mock_exchange.cancel_order.return_value = {"status": "canceled"}
        result = self.executor.cancel_order("order_123", "BTC/USDT")
        assert result is True
        self.mock_exchange.cancel_order.assert_called_once_with("order_123", "BTC/USDT")

    def test_cancel_order_failure(self):
        """Failed cancel should return False."""
        self.mock_exchange.cancel_order.side_effect = ccxt.ExchangeError("Not found")
        result = self.executor.cancel_order("bad_id", "BTC/USDT")
        assert result is False

    def test_cancel_order_default_symbol(self):
        """Cancel without symbol should use Config.TRADING_PAIR."""
        self.mock_exchange.cancel_order.return_value = {"status": "canceled"}
        with patch("executor.Config") as mock_config:
            mock_config.TRADING_PAIR = "ETH/USDT"
            self.executor.cancel_order("order_789")
            self.mock_exchange.cancel_order.assert_called_once_with("order_789", "ETH/USDT")

    def test_executor_stores_exchange_ref(self):
        """LiveExecutor should maintain exchange reference."""
        assert self.executor.exchange is self.mock_exchange


class TestPartialClose:
    def _make_position(self):
        from datetime import UTC, datetime

        from risk_manager import Position
        return Position(
            symbol="BTC/USDT", side="long",
            entry_price=100_000.0, quantity=1.0,
            entry_time=datetime.now(UTC),
            stop_loss=98_000.0, take_profit=105_000.0,
        )

    def test_partial_close_reduces_quantity(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        assert abs(pos.quantity - 0.5) < 1e-9

    def test_partial_close_records_order(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        order = executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        assert order["status"] == "filled"
        assert abs(order["quantity"] - 0.5) < 1e-9
        assert order["reason"] == "partial_tp"

    def test_partial_close_pnl_positive_for_long_above_entry(self):
        from executor import PaperExecutor
        pos = self._make_position()
        executor = PaperExecutor()
        order = executor.partial_close(pos, fraction=0.5, current_price=102_000.0, reason="partial_tp")
        # PnL = (102000 - 100000) * 0.5 qty = 1000
        assert order["pnl"] == pytest.approx(1000.0)


class TestLiveExecutorPartialClose:
    """LiveExecutor.partial_close() — real order via mocked exchange."""

    def _make_position(self, side: str = "long") -> Any:
        from datetime import UTC, datetime

        from risk_manager import Position
        return Position(
            symbol="BTC/USDT",
            side=side,
            entry_price=50_000.0,
            quantity=0.1,
            entry_time=datetime.now(UTC),
            stop_loss=48_000.0,
            take_profit=53_000.0,
        )

    def test_partial_close_places_market_order(self):
        """partial_close() calls exchange.create_market_order with correct qty."""
        import ccxt

        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {
            "id": "live_partial_1",
            "status": "closed",
            "filled": 0.05,
            "average": 51_000.0,
        }
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        executor.partial_close(pos, 0.5, 51_000.0)

        mock_exchange.create_market_order.assert_called_once()
        call_kwargs = mock_exchange.create_market_order.call_args
        assert call_kwargs[0][0] == "BTC/USDT"   # symbol
        assert call_kwargs[0][2] == pytest.approx(0.05)  # qty = 0.1 * 0.5

    def test_partial_close_reduces_position_quantity(self):
        """partial_close() mutates position.quantity by the closed fraction."""
        import ccxt

        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {"id": "x", "status": "closed", "filled": 0.05, "average": 51_000.0}
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        executor.partial_close(pos, 0.5, 51_000.0)

        assert pos.quantity == pytest.approx(0.05)

    def test_partial_close_returns_order_dict(self):
        """partial_close() returns a dict with expected keys."""
        import ccxt

        from executor import LiveExecutor

        mock_exchange = MagicMock(spec=ccxt.Exchange)
        mock_exchange.create_market_order.return_value = {"id": "live_1", "status": "closed", "filled": 0.05, "average": 51_000.0}
        executor = LiveExecutor(mock_exchange)
        pos = self._make_position()

        order = executor.partial_close(pos, 0.5, 51_000.0)

        assert order["status"] == "filled"
        assert order["quantity"] == pytest.approx(0.05)
        assert order["mode"] == "LIVE"


class TestPerpPaperExecutor:
    """PerpPaperExecutor — simulated leveraged futures."""

    def _make_pos_kwargs(self):
        from datetime import UTC, datetime

        return dict(
            symbol="BTC/USDT",
            side="long",
            entry_price=50_000.0,
            quantity=0.1,
            entry_time=datetime.now(UTC),
            stop_loss=48_000.0,
            take_profit=53_000.0,
        )

    def test_place_order_returns_filled(self):
        from executor import PerpPaperExecutor

        ex = PerpPaperExecutor(leverage=3)
        order = ex.place_order("BTC/USDT", "long", 0.1, 50_000.0)
        assert order["status"] == "filled"
        assert order["mode"] == "PERP_PAPER"

    def test_open_position_sets_leverage_fields(self):
        """open_position() creates a Position with leverage and liquidation_price."""
        from executor import PerpPaperExecutor

        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        assert pos.leverage == 3
        assert pos.margin_used == pytest.approx(50_000.0 * 0.1 / 3)
        # Long liq price = entry * (1 - 1/leverage)
        assert pos.liquidation_price == pytest.approx(50_000.0 * (1 - 1 / 3))

    def test_liquidation_price_short(self):
        from datetime import UTC, datetime

        from executor import PerpPaperExecutor

        ex = PerpPaperExecutor(leverage=5)
        kwargs = dict(
            symbol="BTC/USDT",
            side="short",
            entry_price=50_000.0,
            quantity=0.1,
            entry_time=datetime.now(UTC),
            stop_loss=52_000.0,
            take_profit=47_000.0,
        )
        pos = ex.open_position(**kwargs)
        assert pos.liquidation_price == pytest.approx(50_000.0 * (1 + 1 / 5))

    def test_apply_funding_accrues_funding_pnl(self):
        """apply_funding() reduces funding_pnl by rate * notional."""
        from executor import PerpPaperExecutor

        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        ex.apply_funding(pos, funding_rate=0.0001)  # 0.01% per 8h
        # notional = 50_000 * 0.1 = 5000; cost = 5000 * 0.0001 = 0.50
        assert pos.funding_pnl == pytest.approx(-0.50)

    def test_partial_close_works(self):
        from executor import PerpPaperExecutor

        ex = PerpPaperExecutor(leverage=3)
        pos = ex.open_position(**self._make_pos_kwargs())
        order = ex.partial_close(pos, 0.5, 51_000.0)
        assert pos.quantity == pytest.approx(0.05)
        assert order["mode"] == "PERP_PAPER"
