"""
Unit tests for backtester.py — transaction cost modeling, position management,
metric computation, and signal combination.

Tests the Backtester's internal methods without running the full pipeline
(which requires Indicators, ML model, etc.).
"""

import pytest
import numpy as np
import pandas as pd
from datetime import datetime, timedelta

from backtester import Backtester, BacktestTrade, BacktestPosition
from strategies import StrategySignal


@pytest.fixture()
def bt():
    return Backtester(
        initial_capital=10000,
        fee_pct=0.001,
        slippage_pct=0.0005,
        trailing_stop_pct=0.015,
        max_hold_bars=100,
        min_confidence=0.6,
    )


# ---------------------------------------------------------------------------
# Slippage
# ---------------------------------------------------------------------------

class TestSlippage:
    def test_long_entry_slippage_increases_price(self, bt):
        actual = bt._apply_slippage(50000.0, "long", is_entry=True)
        assert actual > 50000.0
        assert actual == pytest.approx(50000.0 * 1.0005)

    def test_long_exit_slippage_decreases_price(self, bt):
        actual = bt._apply_slippage(50000.0, "long", is_entry=False)
        assert actual < 50000.0
        assert actual == pytest.approx(50000.0 * 0.9995)

    def test_short_entry_slippage_decreases_price(self, bt):
        actual = bt._apply_slippage(50000.0, "short", is_entry=True)
        assert actual < 50000.0

    def test_short_exit_slippage_increases_price(self, bt):
        actual = bt._apply_slippage(50000.0, "short", is_entry=False)
        assert actual > 50000.0

    def test_slippage_symmetric(self, bt):
        """Entry and exit slippage should be symmetric for same direction."""
        entry = bt._apply_slippage(50000.0, "long", is_entry=True)
        exit_ = bt._apply_slippage(50000.0, "long", is_entry=False)
        # Entry overpays, exit underpays by same amount
        entry_diff = entry - 50000.0
        exit_diff = 50000.0 - exit_
        assert entry_diff == pytest.approx(exit_diff)


# ---------------------------------------------------------------------------
# Fee calculation
# ---------------------------------------------------------------------------

class TestFees:
    def test_fee_calculation(self, bt):
        fee = bt._calculate_fees(50000.0, 0.01)
        assert fee == pytest.approx(50000.0 * 0.01 * 0.001)

    def test_fee_proportional_to_size(self, bt):
        fee_small = bt._calculate_fees(50000.0, 0.01)
        fee_large = bt._calculate_fees(50000.0, 0.1)
        assert fee_large == pytest.approx(fee_small * 10)


# ---------------------------------------------------------------------------
# Signal combination
# ---------------------------------------------------------------------------

class TestCombine:
    def test_agreement_boosts_confidence(self, bt):
        strat_sig = StrategySignal("BUY", 0.7, "Momentum", "test")
        signal, conf = bt._combine(strat_sig, "BUY", 0.8)
        assert signal == "BUY"
        # Agreement: 0.7*0.6 + 0.8*0.4 + 0.1 = 0.84
        assert conf == pytest.approx(min(0.7 * 0.6 + 0.8 * 0.4 + 0.1, 0.95))

    def test_disagreement_reduces_confidence(self, bt):
        strat_sig = StrategySignal("BUY", 0.7, "Momentum", "test")
        signal, conf = bt._combine(strat_sig, "SELL", 0.8)
        assert signal == "BUY"  # Strategy signal wins
        assert conf == pytest.approx(0.7 * 0.4)  # Heavily dampened

    def test_one_hold_defers_to_active(self, bt):
        strat_sig = StrategySignal("HOLD", 0.5, "Momentum", "test")
        signal, conf = bt._combine(strat_sig, "BUY", 0.8)
        assert signal == "BUY"
        assert conf == pytest.approx(0.8 * 0.6)

    def test_both_hold_returns_hold(self, bt):
        strat_sig = StrategySignal("HOLD", 0.5, "Momentum", "test")
        signal, conf = bt._combine(strat_sig, "HOLD", 0.4)
        assert signal == "HOLD"


# ---------------------------------------------------------------------------
# Position open/close
# ---------------------------------------------------------------------------

class TestPositionManagement:
    def _make_strat_sig(self, signal="BUY"):
        return StrategySignal(
            signal=signal, confidence=0.8,
            strategy_name="Test", reason="test",
            suggested_sl_pct=0.02, suggested_tp_pct=0.05,
        )

    def test_open_position_deducts_capital(self, bt):
        initial = bt.capital
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "trending_up")
        assert bt.capital < initial
        assert len(bt.positions) == 1
        assert bt.positions[0].side == "long"

    def test_open_position_skipped_if_no_capital(self, bt):
        bt.capital = -1.0  # Negative capital (impossible but tests guard)
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "ranging")
        assert len(bt.positions) == 0  # Couldn't afford it

    def test_close_position_records_trade(self, bt):
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "trending_up")
        pos = bt.positions[0]
        bt._close_position(pos, 10, ts + timedelta(hours=10), 51000.0, "take_profit")

        assert len(bt.positions) == 0
        assert len(bt.trades) == 1
        trade = bt.trades[0]
        assert trade.exit_reason == "take_profit"
        assert trade.hold_bars == 10
        assert trade.fees_paid > 0
        assert trade.slippage_cost > 0

    def test_winning_long_trade_has_positive_pnl(self, bt):
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "trending_up")
        pos = bt.positions[0]
        bt._close_position(pos, 5, ts, 52000.0, "take_profit")
        assert bt.trades[0].pnl_gross > 0

    def test_losing_long_trade_has_negative_pnl(self, bt):
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "trending_up")
        pos = bt.positions[0]
        bt._close_position(pos, 5, ts, 48000.0, "stop_loss")
        assert bt.trades[0].pnl_gross < 0

    def test_short_position_opens_correctly(self, bt):
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "SELL", 0.8,
                          self._make_strat_sig("SELL"), "trending_down")
        assert len(bt.positions) == 1
        assert bt.positions[0].side == "short"

    def test_strategy_trades_tracked(self, bt):
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8,
                          self._make_strat_sig("BUY"), "trending_up")
        pos = bt.positions[0]
        bt._close_position(pos, 5, ts, 51000.0, "take_profit")
        assert "Test" in bt.strategy_trades
        assert len(bt.strategy_trades["Test"]) == 1


# ---------------------------------------------------------------------------
# Position exit checking
# ---------------------------------------------------------------------------

class TestCheckPositions:
    def _open_long(self, bt, price=50000.0):
        ts = datetime.now()
        strat = StrategySignal("BUY", 0.8, "Test", "test",
                               suggested_sl_pct=0.02, suggested_tp_pct=0.05)
        bt._open_position(0, ts, price, "BUY", 0.8, strat, "trending_up")

    def test_stop_loss_triggers(self, bt):
        self._open_long(bt, 50000.0)
        row = pd.Series({
            "close": 48000.0, "high": 50000.0, "low": 47500.0,
        })
        bt._check_positions(10, datetime.now(), row)
        assert len(bt.positions) == 0
        assert len(bt.trades) == 1
        assert bt.trades[0].exit_reason == "stop_loss"

    def test_take_profit_triggers(self, bt):
        self._open_long(bt, 50000.0)
        row = pd.Series({
            "close": 53000.0, "high": 54000.0, "low": 52000.0,
        })
        bt._check_positions(10, datetime.now(), row)
        assert len(bt.trades) == 1
        assert bt.trades[0].exit_reason == "take_profit"

    def test_max_duration_triggers(self, bt):
        self._open_long(bt, 50000.0)
        row = pd.Series({
            "close": 50500.0, "high": 50600.0, "low": 50400.0,
        })
        bt._check_positions(100, datetime.now(), row)  # At max hold bars
        assert len(bt.trades) == 1
        assert bt.trades[0].exit_reason == "max_duration"

    def test_trailing_stop_ratchets_up(self, bt):
        """Trailing stop for long positions ratchets up as price rises."""
        # Use wide SL/TP so they don't trigger during the test
        strat = StrategySignal("BUY", 0.8, "Test", "test",
                               suggested_sl_pct=0.10, suggested_tp_pct=0.20)
        ts = datetime.now()
        bt._open_position(0, ts, 50000.0, "BUY", 0.8, strat, "trending_up")
        pos = bt.positions[0]
        initial_trail = pos.trailing_stop

        # Push price up — keep low ABOVE trailing stop level to avoid trigger
        # Trailing after ratchet: 54000*(1-0.015)=53190, so low must be > 53190
        high_row = pd.Series({
            "close": 53500.0, "high": 54000.0, "low": 53500.0,
        })
        bt._check_positions(5, datetime.now(), high_row)
        assert len(bt.positions) == 1  # Still open (within SL/TP)
        assert pos.trailing_stop > initial_trail
        assert pos.highest_price == 54000.0


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------

class TestMetrics:
    def test_no_trades_returns_defaults(self, bt):
        bt.equity_curve = [10000.0, 10000.0]
        result = bt._compute_metrics()
        assert result["total_trades"] == 0
        assert result["win_rate"] == 0

    def test_metrics_with_equity_data(self, bt):
        # Simulate equity curve: up then down
        bt.equity_curve = [10000, 10100, 10200, 10150, 10300]
        # Add a fake trade
        bt.trades.append(BacktestTrade(
            symbol="BTC/USDT", side="long",
            entry_price=50000, exit_price=51000,
            entry_price_actual=50025, exit_price_actual=50975,
            quantity=0.004, pnl_gross=4.0, pnl_net=3.8,
            fees_paid=0.1, slippage_cost=0.1,
            entry_time=datetime.now(), exit_time=datetime.now(),
            exit_reason="take_profit", strategy_name="Momentum",
            regime="trending_up", hold_bars=10,
        ))
        result = bt._compute_metrics()
        assert result["total_return_pct"] > 0
        assert result["total_trades"] == 1
        assert result["win_rate"] == 100.0
        assert result["max_drawdown_pct"] <= 0  # Drawdown is negative or 0
        assert "sharpe_ratio" in result
        assert "sortino_ratio" in result
        assert "total_fees" in result
        assert "exit_reasons" in result

    def test_insufficient_equity_data(self, bt):
        bt.equity_curve = [10000]
        result = bt._compute_metrics()
        assert result.get("error") == "no_data"

    def test_profit_factor_calculation(self, bt):
        bt.equity_curve = [10000, 10100]
        now = datetime.now()
        # 1 winning, 1 losing trade
        bt.trades.append(BacktestTrade(
            symbol="BTC/USDT", side="long",
            entry_price=50000, exit_price=51000,
            entry_price_actual=50025, exit_price_actual=50975,
            quantity=0.004, pnl_gross=4.0, pnl_net=100.0,
            fees_paid=0.1, slippage_cost=0.05,
            entry_time=now, exit_time=now,
            exit_reason="take_profit", strategy_name="Test",
            regime="up", hold_bars=5,
        ))
        bt.trades.append(BacktestTrade(
            symbol="BTC/USDT", side="long",
            entry_price=50000, exit_price=49000,
            entry_price_actual=50025, exit_price_actual=48975,
            quantity=0.004, pnl_gross=-4.0, pnl_net=-50.0,
            fees_paid=0.1, slippage_cost=0.05,
            entry_time=now, exit_time=now,
            exit_reason="stop_loss", strategy_name="Test",
            regime="up", hold_bars=3,
        ))
        result = bt._compute_metrics()
        assert result["profit_factor"] == pytest.approx(100.0 / 50.0)
        assert result["win_rate"] == 50.0

    def test_run_returns_error_on_short_data(self, bt):
        """Backtester.run with < 100 rows returns error."""
        df = pd.DataFrame({
            "open": [50000] * 50, "high": [50100] * 50,
            "low": [49900] * 50, "close": [50000] * 50,
            "volume": [1000] * 50,
        }, index=pd.date_range("2024-01-01", periods=50, freq="h"))
        result = bt.run(df, verbose=False)
        assert result.get("error") == "insufficient_data"
