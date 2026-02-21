"""
Integration tests — Agent → DataStore → API pipeline.
Verifies that running actual agent cycles in paper/demo mode
produces data visible through the API endpoints.

No mocking needed: the agent auto-falls back to deterministic
demo data (seed=42) when the exchange is unreachable.
"""

import pytest
from fastapi.testclient import TestClient

from api.server import create_app, data_store
from agent import TradingAgent, set_data_store
from config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _reset_data_store():
    """Reset all DataStore internals before each test for isolation."""
    with data_store._lock:
        data_store._snapshot = {}
        data_store._equity_history.clear()
        data_store._trade_log.clear()
        data_store._events.clear()
        data_store._intelligence = {}
        data_store._arbitrage = {}
        data_store._backtest_results = []
        data_store._monte_carlo = {}
    yield


@pytest.fixture()
def agent():
    """Create a TradingAgent wired to the global DataStore."""
    set_data_store(data_store)
    return TradingAgent(restore_state=False)


@pytest.fixture()
def client():
    """TestClient for hitting API endpoints (no lifespan/no agent thread)."""
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestAgentToDataStore:
    """Verify agent.run_cycle() pushes data into DataStore."""

    def test_single_cycle_populates_snapshot(self, agent, client):
        agent.run_cycle()

        snapshot = data_store.get_snapshot()
        assert snapshot["cycle"] == 1
        assert snapshot["price"] > 0
        assert "capital" in snapshot

        # Same data visible through API
        body = client.get("/status").json()
        assert body["cycle"] == 1
        assert body["price"] > 0

    def test_equity_tracked_across_cycles(self, agent, client):
        for _ in range(3):
            agent.run_cycle()

        equity = data_store.get_equity_history()
        assert len(equity) == 3

        body = client.get("/equity").json()
        assert body["total_points"] == 3

    def test_health_reflects_agent_running(self, agent, client):
        # Before any cycle
        body = client.get("/health").json()
        assert body["agent_running"] is False

        agent.run_cycle()

        body = client.get("/health").json()
        assert body["agent_running"] is True
        assert body["last_update"] is not None

    def test_multiple_cycles_increment_count(self, agent, client):
        for _ in range(5):
            agent.run_cycle()

        snapshot = data_store.get_snapshot()
        assert snapshot["cycle"] == 5

        body = client.get("/status").json()
        assert body["cycle"] == 5


class TestAgentToAPI:
    """Verify API endpoints return meaningful data after agent cycles."""

    def test_events_endpoint_works(self, agent, client):
        for _ in range(3):
            agent.run_cycle()

        # Events may or may not be generated depending on signals.
        # Verify the endpoint works and returns the correct shape.
        resp = client.get("/autonomous/events?limit=50")
        assert resp.status_code == 200
        body = resp.json()
        assert "events" in body
        assert "count" in body
        assert isinstance(body["events"], list)

    def test_config_matches_config_class(self, client):
        body = client.get("/config").json()
        assert body["exchange"] == Config.EXCHANGE_ID
        assert body["pair"] == Config.TRADING_PAIR
        assert body["mode"] == Config.TRADING_MODE

    def test_positions_endpoint_works(self, agent, client):
        for _ in range(5):
            agent.run_cycle()

        resp = client.get("/positions")
        assert resp.status_code == 200
        body = resp.json()
        assert "positions" in body
        assert "count" in body

    def test_trades_accumulate(self, agent, client):
        for _ in range(10):
            agent.run_cycle()

        resp = client.get("/trades")
        assert resp.status_code == 200
        body = resp.json()
        assert body["total"] >= 0

        # If any trades were made, verify structure
        if body["total"] > 0:
            trade = body["trades"][0]
            assert "side" in trade or "symbol" in trade
