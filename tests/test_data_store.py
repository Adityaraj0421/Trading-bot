"""Tests for the DataStore thread-safe bridge."""
import threading
import pytest

# Add parent to path
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from api.data_store import DataStore


def test_initial_state():
    store = DataStore()
    assert store.get_snapshot() == {}
    assert store.get_equity_history() == []
    assert store.get_trade_log() == []


def test_update_and_get_snapshot():
    store = DataStore()
    store.update_snapshot({"cycle": 1, "price": 95000.0, "capital": 1000.0})
    snapshot = store.get_snapshot()
    assert snapshot["cycle"] == 1
    assert snapshot["price"] == 95000.0
    assert "updated_at" in snapshot


def test_append_equity_point():
    store = DataStore()
    store.append_equity(1000.0, "2026-01-01T00:00:00")
    store.append_equity(1005.0, "2026-01-01T01:00:00")
    history = store.get_equity_history()
    assert len(history) == 2
    assert history[0]["equity"] == 1000.0
    assert history[1]["equity"] == 1005.0


def test_append_trade():
    store = DataStore()
    store.append_trade({"symbol": "BTC/USDT", "side": "long", "pnl_net": 5.0})
    trades = store.get_trade_log()
    assert len(trades) == 1
    assert trades[0]["pnl_net"] == 5.0


def test_append_event():
    store = DataStore()
    store.append_event({"type": "state_change", "description": "normal -> cautious"})
    events = store.get_events()
    assert len(events) == 1


def test_thread_safety():
    store = DataStore()
    errors = []

    def writer(thread_id):
        try:
            for i in range(100):
                store.update_snapshot({"thread": thread_id, "iteration": i})
                store.append_equity(float(i), f"2026-01-01T{i:02d}:00:00")
        except Exception as e:
            errors.append(e)

    threads = [threading.Thread(target=writer, args=(t,)) for t in range(5)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(errors) == 0
    snapshot = store.get_snapshot()
    assert "thread" in snapshot
    history = store.get_equity_history()
    assert len(history) == 500


def test_equity_history_max_size():
    store = DataStore(max_history_size=50)
    for i in range(100):
        store.append_equity(float(i), f"ts_{i}")
    history = store.get_equity_history()
    assert len(history) == 50
    assert history[0]["equity"] == 50.0
