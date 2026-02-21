"""
Tests for TradeDB — SQLite persistence layer.
Covers: trade CRUD, equity snapshots, events, daily summaries, analytics.
Uses tmp_path for database isolation.
"""

import pytest
from trade_db import TradeDB


@pytest.fixture()
def db(tmp_path):
    """Fresh in-directory SQLite database for each test."""
    return TradeDB(db_path=str(tmp_path / "test_trades.db"))


# ── Trade CRUD ────────────────────────────────────────────────────


class TestTradeOperations:
    def test_record_trade_open_returns_id(self, db):
        tid = db.record_trade_open(
            symbol="BTC/USDT", side="long", entry_price=50000,
            quantity=0.1, strategy="momentum", regime="trending_up",
            confidence=0.8, sl=49000, tp=52000, trailing=500,
        )
        assert isinstance(tid, int)
        assert tid > 0

    def test_open_trade_appears_in_list(self, db):
        db.record_trade_open(
            symbol="BTC/USDT", side="long", entry_price=50000,
            quantity=0.1, strategy="momentum", regime="trending_up",
            confidence=0.8, sl=49000, tp=52000, trailing=500,
        )
        trades = db.get_open_trades()
        assert len(trades) == 1
        assert trades[0]["symbol"] == "BTC/USDT"
        assert trades[0]["status"] == "open"

    def test_close_trade_by_id(self, db):
        tid = db.record_trade_open(
            symbol="ETH/USDT", side="short", entry_price=3000,
            quantity=1.0, strategy="reversion", regime="ranging",
            confidence=0.7, sl=3100, tp=2800, trailing=50,
        )
        db.record_trade_close(
            trade_id=tid, exit_price=2900, pnl_gross=100,
            pnl_net=98, fees=2, reason="take_profit", hold_bars=10,
        )
        assert db.get_open_trades() == []
        history = db.get_trade_history(limit=10)
        assert len(history) == 1
        assert history[0]["pnl_net"] == 98

    def test_close_trade_by_symbol_side(self, db):
        db.record_trade_open(
            symbol="BTC/USDT", side="long", entry_price=50000,
            quantity=0.1, strategy="momentum", regime="trending_up",
            confidence=0.8, sl=49000, tp=52000, trailing=500,
        )
        db.record_trade_close(
            symbol="BTC/USDT", side="long", exit_price=51000,
            pnl_gross=100, pnl_net=95, fees=5, reason="trailing_stop",
        )
        assert db.get_open_trades() == []

    def test_multiple_open_trades(self, db):
        for pair in ["BTC/USDT", "ETH/USDT", "SOL/USDT"]:
            db.record_trade_open(
                symbol=pair, side="long", entry_price=100,
                quantity=1.0, strategy="test", regime="ranging",
                confidence=0.5, sl=90, tp=110, trailing=5,
            )
        assert len(db.get_open_trades()) == 3

    def test_trade_history_limit(self, db):
        for i in range(5):
            tid = db.record_trade_open(
                symbol="BTC/USDT", side="long", entry_price=50000 + i,
                quantity=0.1, strategy="test", regime="ranging",
                confidence=0.5, sl=49000, tp=52000, trailing=500,
            )
            db.record_trade_close(trade_id=tid, exit_price=50100 + i, pnl_net=10 + i)
        assert len(db.get_trade_history(limit=3)) == 3
        assert len(db.get_trade_history(limit=100)) == 5

    def test_trade_history_filter_by_strategy(self, db):
        for strat in ["momentum", "reversion", "momentum"]:
            tid = db.record_trade_open(
                symbol="BTC/USDT", side="long", entry_price=50000,
                quantity=0.1, strategy=strat, regime="ranging",
                confidence=0.5, sl=49000, tp=52000, trailing=500,
            )
            db.record_trade_close(trade_id=tid, exit_price=50100, pnl_net=10)
        assert len(db.get_trade_history(strategy="momentum")) == 2
        assert len(db.get_trade_history(strategy="reversion")) == 1


# ── Equity Snapshots ──────────────────────────────────────────────


class TestEquitySnapshots:
    def test_record_and_get_equity(self, db):
        db.record_equity(equity=10000, capital=9500, unrealized_pnl=500,
                         open_positions=2, cycle=1)
        db.record_equity(equity=10100, capital=9600, unrealized_pnl=500,
                         open_positions=2, cycle=2)
        curve = db.get_equity_curve()
        assert len(curve) == 2
        assert curve[0]["cycle"] == 1
        assert curve[1]["equity"] == 10100

    def test_equity_curve_limit(self, db):
        for i in range(10):
            db.record_equity(equity=10000 + i, capital=10000, unrealized_pnl=0,
                             open_positions=0, cycle=i)
        assert len(db.get_equity_curve(limit=5)) == 5


# ── Events ────────────────────────────────────────────────────────


class TestEvents:
    def test_record_and_get_events(self, db):
        db.record_event("trade_open", "Opened BTC long", severity="info")
        db.record_event("error", "API timeout", severity="error", data={"code": 504})
        events = db.get_events()
        assert len(events) == 2

    def test_filter_events_by_type(self, db):
        db.record_event("trade_open", "Opened BTC")
        db.record_event("error", "Timeout")
        db.record_event("trade_close", "Closed BTC")
        assert len(db.get_events(event_type="error")) == 1

    def test_event_data_json(self, db):
        db.record_event("test", "test event", data={"key": "value"})
        events = db.get_events()
        import json
        data = json.loads(events[0]["data"])
        assert data["key"] == "value"


# ── Daily Summaries ───────────────────────────────────────────────


class TestDailySummaries:
    def test_empty_day_returns_zero_trades(self, db):
        summary = db.generate_daily_summary("2025-01-01")
        assert summary["trade_count"] == 0

    def test_cached_summary_returned(self, db):
        # Generate once
        db.generate_daily_summary("2025-01-01")
        # Second call should return cached
        summary = db.generate_daily_summary("2025-01-01")
        assert summary["trade_count"] == 0


# ── Analytics ─────────────────────────────────────────────────────


class TestAnalytics:
    def _populate_trades(self, db):
        """Helper: create a few closed trades for analytics."""
        for i, (strat, regime, pnl) in enumerate([
            ("momentum", "trending_up", 100),
            ("momentum", "trending_up", -30),
            ("reversion", "ranging", 50),
            ("reversion", "ranging", 60),
        ]):
            tid = db.record_trade_open(
                symbol="BTC/USDT", side="long", entry_price=50000,
                quantity=0.1, strategy=strat, regime=regime,
                confidence=0.7, sl=49000, tp=52000, trailing=500,
            )
            db.record_trade_close(
                trade_id=tid, exit_price=50000 + pnl * 10,
                pnl_gross=pnl, pnl_net=pnl - 2, fees=2,
                reason="test", hold_bars=5,
            )

    def test_strategy_performance(self, db):
        self._populate_trades(db)
        perf = db.get_strategy_performance()
        assert "momentum" in perf
        assert "reversion" in perf
        assert perf["momentum"]["total_trades"] == 2

    def test_regime_performance(self, db):
        self._populate_trades(db)
        perf = db.get_regime_performance()
        assert "trending_up" in perf
        assert "ranging" in perf

    def test_total_stats(self, db):
        self._populate_trades(db)
        stats = db.get_total_stats()
        assert stats["total_trades"] == 4
        assert "win_rate" in stats
        assert stats["wins"] == 3  # 100, 50, 60 are positive

    def test_empty_stats(self, db):
        stats = db.get_total_stats()
        assert stats.get("total_trades", 0) == 0


class TestDBClose:
    def test_close_is_safe(self, db):
        """close() is a no-op but shouldn't raise."""
        db.close()
