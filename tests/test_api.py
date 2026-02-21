"""
Tests for the FastAPI server and all route modules.
Uses FastAPI's TestClient (backed by httpx) to exercise every endpoint
and verify response shapes, status codes, and data visibility.
"""

import pytest
from fastapi.testclient import TestClient

from api.data_store import DataStore
from api.routes import status, trading, autonomous, backtest, intelligence, arbitrage, risk
from api.server import create_app, data_store
from config import Config


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def client():
    """Return a TestClient wired to the app (no lifespan / no agent thread).

    When API_AUTH_KEY is set in .env, the middleware rejects unauthenticated
    requests.  We inject the key into default headers so all tests pass
    regardless of whether auth is enabled.
    """
    app = create_app()
    c = TestClient(app)
    if Config.API_AUTH_KEY:
        c.headers["X-API-Key"] = Config.API_AUTH_KEY
    return c


@pytest.fixture(autouse=True)
def _reset_data_store():
    """
    Reset the global data_store before each test so tests are isolated.
    We mutate the internals rather than replacing the object because the
    routers already hold a reference to it.
    """
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


# =========================================================================
# Status routes
# =========================================================================

class TestHealthEndpoint:
    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_shape_no_agent(self, client):
        body = client.get("/health").json()
        assert body["status"] == "healthy"
        assert body["agent_running"] is False
        assert body["last_update"] is None

    def test_health_agent_running_after_snapshot(self, client):
        data_store.update_snapshot({"cycle": 1})
        body = client.get("/health").json()
        assert body["agent_running"] is True
        assert body["last_update"] is not None


class TestStatusEndpoint:
    def test_status_waiting(self, client):
        body = client.get("/status").json()
        assert body["status"] == "waiting"

    def test_status_returns_snapshot(self, client):
        data_store.update_snapshot({"cycle": 42, "equity": 1050.0})
        body = client.get("/status").json()
        assert body["cycle"] == 42
        assert body["equity"] == 1050.0
        assert "updated_at" in body


class TestConfigEndpoint:
    def test_config_returns_200(self, client):
        resp = client.get("/config")
        assert resp.status_code == 200

    def test_config_contains_expected_keys(self, client):
        body = client.get("/config").json()
        expected_keys = {
            "exchange", "pair", "pairs", "timeframe",
            "confirmation_timeframe", "mode", "initial_capital",
            "max_position_pct", "stop_loss_pct", "take_profit_pct",
            "trailing_stop_pct", "fee_pct", "slippage_pct",
            "agent_interval_seconds",
        }
        assert expected_keys.issubset(body.keys())


# =========================================================================
# Trading routes
# =========================================================================

class TestTradesEndpoint:
    def test_trades_empty(self, client):
        body = client.get("/trades").json()
        assert body == {"trades": [], "total": 0}

    def test_trades_populated(self, client):
        data_store.append_trade({"side": "buy", "price": 50000})
        data_store.append_trade({"side": "sell", "price": 51000})
        body = client.get("/trades").json()
        assert body["total"] == 2
        assert len(body["trades"]) == 2

    def test_trades_limit(self, client):
        for i in range(10):
            data_store.append_trade({"id": i})
        body = client.get("/trades?limit=3").json()
        assert body["total"] == 10
        assert len(body["trades"]) == 3
        # Should return the last 3
        assert body["trades"][0]["id"] == 7


class TestPositionsEndpoint:
    def test_positions_empty(self, client):
        body = client.get("/positions").json()
        assert body == {"positions": [], "count": 0}

    def test_positions_from_snapshot(self, client):
        data_store.update_snapshot({
            "positions": [
                {"pair": "BTC/USDT", "side": "long", "size": 0.1},
            ]
        })
        body = client.get("/positions").json()
        assert body["count"] == 1
        assert body["positions"][0]["pair"] == "BTC/USDT"


class TestEquityEndpoint:
    def test_equity_empty(self, client):
        body = client.get("/equity").json()
        assert body == {"equity": [], "total_points": 0}

    def test_equity_populated(self, client):
        data_store.append_equity(1000.0, "2025-01-01T00:00:00")
        data_store.append_equity(1010.0, "2025-01-01T01:00:00")
        data_store.append_equity(1005.0, "2025-01-01T02:00:00")
        body = client.get("/equity").json()
        assert body["total_points"] == 3
        assert len(body["equity"]) == 3

    def test_equity_limit(self, client):
        for i in range(5):
            data_store.append_equity(1000.0 + i, f"2025-01-01T0{i}:00:00")
        body = client.get("/equity?limit=2").json()
        assert body["total_points"] == 5
        assert len(body["equity"]) == 2


# =========================================================================
# Autonomous routes
# =========================================================================

class TestAutonomousStatus:
    def test_autonomous_not_running(self, client):
        body = client.get("/autonomous/status").json()
        assert body["status"] == "not_running"

    def test_autonomous_running(self, client):
        data_store.update_snapshot({
            "autonomous": {"mode": "aggressive", "uptime": 3600}
        })
        body = client.get("/autonomous/status").json()
        # Route returns the autonomous dict directly when populated
        assert body["mode"] == "aggressive"
        assert body["uptime"] == 3600


class TestAutonomousEvents:
    def test_events_empty(self, client):
        body = client.get("/autonomous/events").json()
        assert body == {"events": [], "count": 0}

    def test_events_populated(self, client):
        data_store.append_event({"type": "trade", "msg": "bought BTC"})
        data_store.append_event({"type": "rebalance", "msg": "rebalanced"})
        body = client.get("/autonomous/events?limit=50").json()
        assert body["count"] == 2

    def test_events_limit(self, client):
        for i in range(10):
            data_store.append_event({"id": i})
        body = client.get("/autonomous/events?limit=3").json()
        assert body["count"] == 3


# =========================================================================
# Backtest routes (Phase 4 — real implementation)
# =========================================================================

class TestBacktest:
    def test_run_starts_background(self, client):
        body = client.post("/backtest/run", json={"scenario": "bull_run"}).json()
        assert body["status"] == "started"

    def test_results_empty(self, client):
        body = client.get("/backtest/results").json()
        assert body == {"results": [], "total": 0}

    def test_results_with_data(self, client):
        data_store.update_backtest_results([{"strategy": "sma", "sharpe": 1.5}])
        body = client.get("/backtest/results").json()
        assert body["total"] == 1
        assert body["results"][0]["sharpe"] == 1.5

    def test_scenarios_list(self, client):
        body = client.get("/backtest/scenarios").json()
        assert "scenarios" in body
        assert "bull_run" in body["scenarios"]
        assert len(body["scenarios"]) == 6


# =========================================================================
# Intelligence routes (Phase 5 — real implementation)
# =========================================================================

class TestIntelligence:
    def test_signals_not_enabled(self, client):
        from unittest.mock import patch
        with patch("config.Config.any_intelligence_enabled", return_value=False):
            body = client.get("/intelligence/signals").json()
        assert body["status"] == "not_enabled"

    def test_signals_with_data(self, client):
        data_store.update_intelligence({"whale_alert": True, "sentiment": 0.7})
        body = client.get("/intelligence/signals").json()
        assert body["whale_alert"] is True
        assert body["sentiment"] == 0.7


# =========================================================================
# Arbitrage routes (Phase 6 — real implementation)
# =========================================================================

class TestArbitrage:
    def test_opportunities_not_enabled(self, client):
        body = client.get("/arbitrage/opportunities").json()
        assert body["status"] == "not_enabled"

    def test_opportunities_with_data(self, client):
        data_store.update_arbitrage({"spread": 0.5, "pair": "BTC/USDT"})
        body = client.get("/arbitrage/opportunities").json()
        assert body["spread"] == 0.5
        assert body["pair"] == "BTC/USDT"


# =========================================================================
# Risk routes (Phase 7 — real implementation)
# =========================================================================

class TestRisk:
    def test_simulation_not_run(self, client):
        body = client.get("/risk/simulation").json()
        assert body["status"] == "not_run"

    def test_simulation_with_data(self, client):
        data_store.update_monte_carlo({"var_95": -0.05, "simulations": 10000, "status": "completed"})
        body = client.get("/risk/simulation").json()
        assert body["status"] == "completed"
        assert body["var_95"] == -0.05

    def test_monte_carlo_starts_background(self, client):
        body = client.post("/risk/monte-carlo", json={}).json()
        assert body["status"] == "started"


# =========================================================================
# Cross-cutting: data visibility through endpoints
# =========================================================================

class TestDataVisibility:
    """Verify that pushing data into the global data_store is visible
    through the API endpoints (i.e., the store reference is shared)."""

    def test_snapshot_visible_in_health_and_status(self, client):
        # Before: no data
        assert client.get("/health").json()["agent_running"] is False
        assert client.get("/status").json()["status"] == "waiting"

        # Push snapshot
        data_store.update_snapshot({"cycle": 1, "equity": 1000.0})

        # After: data visible
        assert client.get("/health").json()["agent_running"] is True
        status_body = client.get("/status").json()
        assert status_body["cycle"] == 1
        assert status_body["equity"] == 1000.0

    def test_trade_then_equity_then_event_pipeline(self, client):
        data_store.append_trade({"side": "buy", "price": 42000})
        data_store.append_equity(1000.0, "2025-01-01T00:00:00")
        data_store.append_event({"type": "signal", "msg": "bullish crossover"})

        assert client.get("/trades").json()["total"] == 1
        assert client.get("/equity").json()["total_points"] == 1
        assert client.get("/autonomous/events").json()["count"] == 1
