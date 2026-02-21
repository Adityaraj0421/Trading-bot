# Crypto Trading Agent v6.0 Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Transform the v5.0 autonomous trading agent into a full-stack production platform with FastAPI backend, React dashboard, Docker deployment, extended backtesting, intelligence signals, multi-exchange arbitrage, and Monte Carlo risk simulation.

**Architecture:** Agent runs in a background thread, pushes cycle snapshots to a thread-safe DataStore. FastAPI reads from the DataStore and serves JSON to a Next.js dashboard that polls every 5 seconds. Docker Compose orchestrates everything. Intelligence signals, arbitrage, and risk simulation plug into the agent cycle and expose data via the same API.

**Tech Stack:** Python 3.11 (FastAPI, uvicorn, yfinance, textblob), Next.js 14 (React, Tailwind CSS, Recharts, SWR), Docker Compose, CCXT (Binance/Coinbase/Kraken).

---

## Phase 1: DataStore + FastAPI (Foundation)

### Task 1.1: Update Config with API and Data Directory Settings

**Files:**
- Modify: `/Users/adityaraj0421/Finances/crypto-trading-agent/config.py`

**Step 1: Add new config settings to config.py**

Add these settings after the existing `LOG_LEVEL` line (line 54):

```python
    # API Server
    API_HOST = os.getenv("API_HOST", "0.0.0.0")
    API_PORT = int(os.getenv("API_PORT", "8000"))
    ENABLE_API = os.getenv("ENABLE_API", "true").lower() == "true"

    # Data directory (for Docker volume mounts)
    DATA_DIR = os.getenv("DATA_DIR", ".")

    # Multi-exchange (for arbitrage)
    BINANCE_API_KEY = os.getenv("BINANCE_API_KEY", "")
    BINANCE_API_SECRET = os.getenv("BINANCE_API_SECRET", "")
    COINBASE_API_KEY = os.getenv("COINBASE_API_KEY", "")
    COINBASE_API_SECRET = os.getenv("COINBASE_API_SECRET", "")
    KRAKEN_API_KEY = os.getenv("KRAKEN_API_KEY", "")
    KRAKEN_API_SECRET = os.getenv("KRAKEN_API_SECRET", "")

    # Intelligence toggles
    ENABLE_ONCHAIN = os.getenv("ENABLE_ONCHAIN", "false").lower() == "true"
    ENABLE_WHALE_TRACKING = os.getenv("ENABLE_WHALE_TRACKING", "false").lower() == "true"
    ENABLE_NEWS_NLP = os.getenv("ENABLE_NEWS_NLP", "false").lower() == "true"
    ENABLE_CORRELATION = os.getenv("ENABLE_CORRELATION", "false").lower() == "true"
    ENABLE_ORDERBOOK = os.getenv("ENABLE_ORDERBOOK", "false").lower() == "true"

    # Arbitrage
    ARBITRAGE_ENABLED = os.getenv("ARBITRAGE_ENABLED", "false").lower() == "true"
    ARBITRAGE_MIN_SPREAD_PCT = float(os.getenv("ARBITRAGE_MIN_SPREAD_PCT", "0.003"))

    # Monte Carlo
    MC_SIMULATIONS = int(os.getenv("MC_SIMULATIONS", "10000"))
    MC_HORIZON_DAYS = int(os.getenv("MC_HORIZON_DAYS", "252"))
```

Also modify `STATE_FILE` and `LOG_FILE` to use `DATA_DIR`:

Change line 50-54 from:
```python
    STATE_FILE = os.getenv("STATE_FILE", "agent_state.json")
    LOG_FILE = os.getenv("LOG_FILE", "agent.log")
```
to:
```python
    STATE_FILE = os.getenv("STATE_FILE", None)  # Set in _resolve_paths
    LOG_FILE = os.getenv("LOG_FILE", None)

    @classmethod
    def _resolve_paths(cls):
        """Resolve file paths relative to DATA_DIR."""
        if cls.STATE_FILE is None:
            cls.STATE_FILE = os.path.join(cls.DATA_DIR, "agent_state.json")
        if cls.LOG_FILE is None:
            cls.LOG_FILE = os.path.join(cls.DATA_DIR, "agent.log")
```

And call `cls._resolve_paths()` at the start of the `validate()` method.

**Step 2: Update .env.example**

Append to `/Users/adityaraj0421/Finances/crypto-trading-agent/.env.example`:

```
# API Server
API_HOST=0.0.0.0
API_PORT=8000
ENABLE_API=true

# Data directory (state, logs, models)
DATA_DIR=.

# Intelligence (all free APIs)
ENABLE_ONCHAIN=false
ENABLE_WHALE_TRACKING=false
ENABLE_NEWS_NLP=false
ENABLE_CORRELATION=false
ENABLE_ORDERBOOK=false

# Multi-exchange (for arbitrage)
ARBITRAGE_ENABLED=false
# BINANCE_API_KEY=
# BINANCE_API_SECRET=
# COINBASE_API_KEY=
# COINBASE_API_SECRET=
# KRAKEN_API_KEY=
# KRAKEN_API_SECRET=

# Monte Carlo
MC_SIMULATIONS=10000
MC_HORIZON_DAYS=252
```

**Step 3: Commit**

```bash
git add config.py .env.example
git commit -m "feat: add API, intelligence, arbitrage, and MC config settings"
```

---

### Task 1.2: Create the DataStore (Thread-Safe Agent↔API Bridge)

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/__init__.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/data_store.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/tests/test_data_store.py`

**Step 1: Write the test file**

```python
"""Tests for the DataStore thread-safe bridge."""
import threading
import time
import pytest
from api.data_store import DataStore


def test_initial_state():
    store = DataStore()
    snapshot = store.get_snapshot()
    assert snapshot == {}
    assert store.get_equity_history() == []
    assert store.get_trade_log() == []


def test_update_and_get_snapshot():
    store = DataStore()
    store.update_snapshot({
        "cycle": 1,
        "price": 95000.0,
        "capital": 1000.0,
        "total_pnl": 0.0,
    })
    snapshot = store.get_snapshot()
    assert snapshot["cycle"] == 1
    assert snapshot["price"] == 95000.0


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
    trade = {"symbol": "BTC/USDT", "side": "long", "pnl_net": 5.0}
    store.append_trade(trade)
    trades = store.get_trade_log()
    assert len(trades) == 1
    assert trades[0]["pnl_net"] == 5.0


def test_append_event():
    store = DataStore()
    store.append_event({"type": "state_change", "description": "normal -> cautious"})
    events = store.get_events()
    assert len(events) == 1


def test_thread_safety():
    """Multiple threads writing simultaneously should not corrupt data."""
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
    # Snapshot should exist (from any thread)
    snapshot = store.get_snapshot()
    assert "thread" in snapshot
    # Should have equity points from all threads
    history = store.get_equity_history()
    assert len(history) == 500  # 5 threads × 100 iterations


def test_equity_history_max_size():
    """Equity history should cap at max_history_size."""
    store = DataStore(max_history_size=50)
    for i in range(100):
        store.append_equity(float(i), f"ts_{i}")
    history = store.get_equity_history()
    assert len(history) == 50
    assert history[0]["equity"] == 50.0  # Oldest entries trimmed
```

**Step 2: Run test to verify it fails**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
python -m pytest tests/test_data_store.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'api'`

**Step 3: Create api/__init__.py**

```python
"""API package for the Crypto Trading Agent."""
```

**Step 4: Create api/data_store.py**

```python
"""
DataStore — Thread-Safe Agent↔API Bridge
==========================================
The agent writes cycle snapshots, equity points, trades, and events.
The API reads them. All operations are thread-safe via a single lock.
"""

import threading
from collections import deque
from datetime import datetime
from typing import Any


class DataStore:
    """
    Thread-safe in-memory data bridge between the trading agent and API.

    The agent calls update_snapshot(), append_equity(), append_trade(),
    and append_event() from its main loop. The API reads via get_*() methods.
    """

    def __init__(self, max_history_size: int = 10000):
        self._lock = threading.Lock()
        self._snapshot: dict = {}
        self._equity_history: deque = deque(maxlen=max_history_size)
        self._trade_log: list[dict] = []
        self._events: deque = deque(maxlen=1000)
        self._intelligence: dict = {}
        self._arbitrage: dict = {}
        self._backtest_results: list[dict] = []
        self._monte_carlo: dict = {}

    # --- Writer methods (called by agent) ---

    def update_snapshot(self, snapshot: dict) -> None:
        """Replace the current cycle snapshot."""
        with self._lock:
            self._snapshot = snapshot.copy()
            self._snapshot["updated_at"] = datetime.now().isoformat()

    def append_equity(self, equity: float, timestamp: str) -> None:
        """Add an equity data point for the chart."""
        with self._lock:
            self._equity_history.append({
                "equity": equity,
                "timestamp": timestamp,
            })

    def append_trade(self, trade: dict) -> None:
        """Record a completed trade."""
        with self._lock:
            self._trade_log.append(trade)

    def append_event(self, event: dict) -> None:
        """Record an autonomous event."""
        with self._lock:
            self._events.append(event)

    def update_intelligence(self, signals: dict) -> None:
        """Update intelligence signals snapshot."""
        with self._lock:
            self._intelligence = signals.copy()

    def update_arbitrage(self, data: dict) -> None:
        """Update arbitrage opportunities snapshot."""
        with self._lock:
            self._arbitrage = data.copy()

    def update_backtest_results(self, results: list[dict]) -> None:
        """Store backtest results."""
        with self._lock:
            self._backtest_results = results

    def update_monte_carlo(self, results: dict) -> None:
        """Store Monte Carlo simulation results."""
        with self._lock:
            self._monte_carlo = results.copy()

    # --- Reader methods (called by API) ---

    def get_snapshot(self) -> dict:
        """Get the latest cycle snapshot."""
        with self._lock:
            return self._snapshot.copy()

    def get_equity_history(self) -> list[dict]:
        """Get full equity time series."""
        with self._lock:
            return list(self._equity_history)

    def get_trade_log(self) -> list[dict]:
        """Get all completed trades."""
        with self._lock:
            return self._trade_log.copy()

    def get_events(self, limit: int = 100) -> list[dict]:
        """Get recent autonomous events."""
        with self._lock:
            events = list(self._events)
            return events[-limit:]

    def get_intelligence(self) -> dict:
        """Get latest intelligence signals."""
        with self._lock:
            return self._intelligence.copy()

    def get_arbitrage(self) -> dict:
        """Get latest arbitrage data."""
        with self._lock:
            return self._arbitrage.copy()

    def get_backtest_results(self) -> list[dict]:
        """Get stored backtest results."""
        with self._lock:
            return self._backtest_results.copy()

    def get_monte_carlo(self) -> dict:
        """Get Monte Carlo results."""
        with self._lock:
            return self._monte_carlo.copy()
```

**Step 5: Run tests to verify they pass**

```bash
python -m pytest tests/test_data_store.py -v
```
Expected: All 7 tests PASS

**Step 6: Commit**

```bash
git add api/__init__.py api/data_store.py tests/test_data_store.py
git commit -m "feat: add thread-safe DataStore for agent↔API bridge"
```

---

### Task 1.3: Create FastAPI Server with Status Routes

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/__init__.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/status.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/trading.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/autonomous.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/server.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/tests/test_api.py`
- Modify: `/Users/adityaraj0421/Finances/crypto-trading-agent/requirements.txt`

**Step 1: Add dependencies to requirements.txt**

Append to requirements.txt:

```
fastapi>=0.109.0
uvicorn[standard]>=0.27.0
httpx>=0.27.0
```

**Step 2: Install dependencies**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
pip install fastapi "uvicorn[standard]" httpx
```

**Step 3: Create api/routes/__init__.py**

```python
"""API route modules."""
```

**Step 4: Create api/routes/status.py**

```python
"""Status and health check routes."""

from fastapi import APIRouter
from api.data_store import DataStore

router = APIRouter(tags=["status"])


def create_router(store: DataStore) -> APIRouter:
    """Create status router with DataStore dependency."""

    @router.get("/health")
    async def health():
        snapshot = store.get_snapshot()
        return {
            "status": "healthy",
            "agent_running": bool(snapshot),
            "last_update": snapshot.get("updated_at"),
        }

    @router.get("/status")
    async def status():
        snapshot = store.get_snapshot()
        if not snapshot:
            return {"status": "waiting", "message": "Agent has not completed a cycle yet"}
        return snapshot

    @router.get("/config")
    async def config():
        from config import Config
        return {
            "exchange": Config.EXCHANGE_ID,
            "trading_pair": Config.TRADING_PAIR,
            "timeframe": Config.TIMEFRAME,
            "trading_mode": Config.TRADING_MODE,
            "initial_capital": Config.INITIAL_CAPITAL,
            "max_position_pct": Config.MAX_POSITION_PCT,
            "stop_loss_pct": Config.STOP_LOSS_PCT,
            "take_profit_pct": Config.TAKE_PROFIT_PCT,
            "trailing_stop_pct": Config.TRAILING_STOP_PCT,
            "fee_pct": Config.FEE_PCT,
        }

    return router
```

**Step 5: Create api/routes/trading.py**

```python
"""Trading data routes — trades, positions, equity."""

from fastapi import APIRouter, Query
from api.data_store import DataStore

router = APIRouter(tags=["trading"])


def create_router(store: DataStore) -> APIRouter:

    @router.get("/trades")
    async def trades(limit: int = Query(100, ge=1, le=1000)):
        trade_log = store.get_trade_log()
        return {"trades": trade_log[-limit:], "total": len(trade_log)}

    @router.get("/positions")
    async def positions():
        snapshot = store.get_snapshot()
        return {
            "positions": snapshot.get("positions", []),
            "count": len(snapshot.get("positions", [])),
        }

    @router.get("/equity")
    async def equity(limit: int = Query(0, ge=0)):
        history = store.get_equity_history()
        if limit > 0:
            history = history[-limit:]
        return {"equity": history, "total_points": len(store.get_equity_history())}

    return router
```

**Step 6: Create api/routes/autonomous.py**

```python
"""Autonomous system status routes."""

from fastapi import APIRouter, Query
from api.data_store import DataStore

router = APIRouter(prefix="/autonomous", tags=["autonomous"])


def create_router(store: DataStore) -> APIRouter:

    @router.get("/status")
    async def autonomous_status():
        snapshot = store.get_snapshot()
        return snapshot.get("autonomous", {
            "state": "unknown",
            "message": "No autonomous data yet",
        })

    @router.get("/events")
    async def autonomous_events(limit: int = Query(50, ge=1, le=500)):
        events = store.get_events(limit=limit)
        return {"events": events, "count": len(events)}

    return router
```

**Step 7: Create api/server.py**

```python
"""
FastAPI Server for Crypto Trading Agent
=========================================
Runs alongside the trading agent, serving data to the React dashboard.
The agent runs in a background thread; the API runs in the main thread.
"""

import threading
import sys
import os

# Add parent directory to path so we can import agent modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.data_store import DataStore
from api.routes import status, trading, autonomous


# Global DataStore instance — shared between agent thread and API
data_store = DataStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Start the trading agent in a background thread on server startup."""
    print("[API] Starting trading agent in background thread...")

    agent_thread = threading.Thread(
        target=_run_agent,
        daemon=True,
        name="trading-agent",
    )
    agent_thread.start()
    yield
    print("[API] Shutting down...")


def _run_agent():
    """Run the trading agent in a background thread, pushing data to the store."""
    try:
        from agent import TradingAgent
        agent = TradingAgent()

        # Patch run_cycle to push data to the store after each cycle
        original_run_cycle = agent.run_cycle

        def patched_run_cycle():
            original_run_cycle()
            _push_to_store(agent)

        agent.run_cycle = patched_run_cycle
        agent.run()
    except Exception as e:
        print(f"[API] Agent thread error: {e}")
        import traceback
        traceback.print_exc()


def _push_to_store(agent):
    """Extract data from agent and push to DataStore."""
    try:
        summary = agent.risk.get_summary()
        price = agent._last_price or 0.0
        regime = agent._last_regime.value if agent._last_regime else "unknown"

        # Main snapshot
        data_store.update_snapshot({
            "cycle": agent.cycle_count,
            "price": price,
            "pair": agent.fetcher.exchange.id if not agent.fetcher.using_demo else "demo",
            "trading_pair": agent.risk.positions[0].symbol if agent.risk.positions else "BTC/USDT",
            "capital": summary["capital"],
            "total_pnl": summary["total_pnl"],
            "daily_pnl": summary["daily_pnl"],
            "total_fees": summary["total_fees"],
            "win_rate": summary["win_rate"],
            "total_trades": summary["total_trades"],
            "open_positions": summary["open_positions"],
            "regime": regime,
            "positions": [
                {
                    "symbol": p.symbol,
                    "side": p.side,
                    "entry_price": p.entry_price,
                    "quantity": p.quantity,
                    "unrealized_pnl": p.unrealized_pnl(price),
                    "stop_loss": p.stop_loss,
                    "take_profit": p.take_profit,
                    "trailing_stop": p.trailing_stop,
                    "strategy": p.strategy_name,
                }
                for p in agent.risk.positions
            ],
            "autonomous": agent.decision.get_autonomous_status(),
        })

        # Equity point
        equity_value = summary["capital"] + sum(
            p.unrealized_pnl(price) for p in agent.risk.positions
        )
        from datetime import datetime
        data_store.append_equity(
            round(equity_value, 2),
            datetime.now().isoformat(),
        )

        # New closed trades since last push
        # (We track by count — if trade_history grew, push new ones)
        trade_count = len(agent.risk.trade_history)
        existing_count = len(data_store.get_trade_log())
        if trade_count > existing_count:
            for t in agent.risk.trade_history[existing_count:]:
                data_store.append_trade({
                    "symbol": t.symbol,
                    "side": t.side,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": t.quantity,
                    "pnl_gross": round(t.pnl_gross, 4),
                    "pnl_net": round(t.pnl_net, 4),
                    "fees_paid": round(t.fees_paid, 4),
                    "entry_time": t.entry_time.isoformat(),
                    "exit_time": t.exit_time.isoformat(),
                    "exit_reason": t.exit_reason,
                    "strategy": t.strategy_name,
                    "hold_bars": t.hold_bars,
                })

        # Autonomous events
        for event in list(agent.decision.event_log)[-5:]:
            data_store.append_event(event.to_dict())

    except Exception as e:
        print(f"[API] Error pushing to store: {e}")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="Crypto Trading Agent API",
        description="Real-time data from the autonomous trading agent",
        version="6.0",
        lifespan=lifespan,
    )

    # CORS — allow the dashboard to access the API
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(status.create_router(data_store))
    app.include_router(trading.create_router(data_store))
    app.include_router(autonomous.create_router(data_store))

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    from config import Config
    uvicorn.run(
        "api.server:app",
        host=Config.API_HOST,
        port=Config.API_PORT,
        reload=False,
    )
```

**Step 8: Write API tests**

```python
"""Tests for the FastAPI endpoints."""
import pytest
from fastapi.testclient import TestClient
from api.data_store import DataStore
from api.server import create_app, data_store


@pytest.fixture
def client():
    """Create a test client with a clean DataStore."""
    app = create_app()
    return TestClient(app)


def test_health_endpoint_empty(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["agent_running"] is False


def test_status_endpoint_empty(client):
    response = client.get("/status")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "waiting"


def test_config_endpoint(client):
    response = client.get("/config")
    assert response.status_code == 200
    data = response.json()
    assert "exchange" in data
    assert "trading_pair" in data


def test_trades_endpoint_empty(client):
    response = client.get("/trades")
    assert response.status_code == 200
    data = response.json()
    assert data["trades"] == []
    assert data["total"] == 0


def test_equity_endpoint_empty(client):
    response = client.get("/equity")
    assert response.status_code == 200
    data = response.json()
    assert data["equity"] == []


def test_positions_endpoint_empty(client):
    response = client.get("/positions")
    assert response.status_code == 200
    data = response.json()
    assert data["positions"] == []


def test_autonomous_status_empty(client):
    response = client.get("/autonomous/status")
    assert response.status_code == 200
    data = response.json()
    assert "state" in data


def test_autonomous_events_empty(client):
    response = client.get("/autonomous/events")
    assert response.status_code == 200
    data = response.json()
    assert data["events"] == []


def test_health_with_data(client):
    """Push data to store, then check health."""
    data_store.update_snapshot({"cycle": 5, "price": 95000.0})
    response = client.get("/health")
    data = response.json()
    assert data["agent_running"] is True


def test_trades_with_data(client):
    data_store.append_trade({"symbol": "BTC/USDT", "pnl_net": 15.0})
    response = client.get("/trades")
    data = response.json()
    assert data["total"] >= 1
```

**Step 9: Run tests**

```bash
python -m pytest tests/test_api.py -v
```
Expected: All tests PASS

**Step 10: Commit**

```bash
git add api/ tests/test_api.py requirements.txt
git commit -m "feat: add FastAPI server with status, trading, and autonomous routes"
```

---

### Task 1.4: Integrate DataStore into Agent

**Files:**
- Modify: `/Users/adityaraj0421/Finances/crypto-trading-agent/agent.py`

**Step 1: Add DataStore import and optional injection**

At the top of agent.py, after the existing imports (after line 42), add:

```python
# Optional API integration
_data_store = None

def set_data_store(store):
    """Allow the API server to inject a DataStore instance."""
    global _data_store
    _data_store = store
```

**Step 2: Add push method to TradingAgent**

After `_print_portfolio` method (after line 442), add:

```python
    def _push_to_data_store(self):
        """Push cycle snapshot to DataStore if available."""
        if _data_store is None:
            return
        try:
            summary = self.risk.get_summary()
            price = self._last_price or 0.0
            regime = self._last_regime.value if self._last_regime else "unknown"

            _data_store.update_snapshot({
                "cycle": self.cycle_count,
                "price": price,
                "pair": Config.EXCHANGE_ID,
                "trading_pair": Config.TRADING_PAIR,
                "capital": summary["capital"],
                "total_pnl": summary["total_pnl"],
                "daily_pnl": summary["daily_pnl"],
                "total_fees": summary["total_fees"],
                "win_rate": summary["win_rate"],
                "total_trades": summary["total_trades"],
                "open_positions": summary["open_positions"],
                "regime": regime,
                "positions": [
                    {
                        "symbol": p.symbol,
                        "side": p.side,
                        "entry_price": p.entry_price,
                        "quantity": p.quantity,
                        "unrealized_pnl": p.unrealized_pnl(price),
                        "stop_loss": p.stop_loss,
                        "take_profit": p.take_profit,
                        "trailing_stop": p.trailing_stop,
                        "strategy": p.strategy_name,
                    }
                    for p in self.risk.positions
                ],
                "autonomous": self.decision.get_autonomous_status(),
            })

            # Equity point
            equity_value = summary["capital"] + sum(
                p.unrealized_pnl(price) for p in self.risk.positions
            )
            from datetime import datetime as dt
            _data_store.append_equity(
                round(equity_value, 2),
                dt.now().isoformat(),
            )
        except Exception as e:
            pass  # Don't let data store errors crash the agent
```

**Step 3: Call _push_to_data_store at end of run_cycle**

In the `run_cycle` method, after line 344 (`if self.cycle_count % 10 == 0: self.state_mgr.save(self)`), add:

```python
        # Push to API data store
        self._push_to_data_store()
```

**Step 4: Simplify api/server.py to use injection**

Update `_run_agent()` in api/server.py to use the cleaner injection pattern:

```python
def _run_agent():
    """Run the trading agent in a background thread, pushing data to the store."""
    try:
        from agent import TradingAgent, set_data_store
        set_data_store(data_store)
        agent = TradingAgent()
        agent.run()
    except Exception as e:
        print(f"[API] Agent thread error: {e}")
        import traceback
        traceback.print_exc()
```

Remove the `_push_to_store` function and the monkey-patching code from server.py (the agent now handles this internally).

**Step 5: Verify agent still runs standalone**

```bash
python agent.py 2
```
Expected: Agent runs 2 cycles normally, no errors about DataStore

**Step 6: Commit**

```bash
git add agent.py api/server.py
git commit -m "feat: integrate DataStore into agent cycle with injection pattern"
```

---

### Task 1.5: Add Stub Routes for Later Phases

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/backtest.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/intelligence.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/arbitrage.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/routes/risk.py`
- Modify: `/Users/adityaraj0421/Finances/crypto-trading-agent/api/server.py` (register new routers)

**Step 1: Create stub route files**

Each returns a "coming soon" response so the dashboard can render placeholder pages.

`api/routes/backtest.py`:
```python
"""Backtest routes — run and view backtest results."""
from fastapi import APIRouter
from api.data_store import DataStore

router = APIRouter(prefix="/backtest", tags=["backtest"])

def create_router(store: DataStore) -> APIRouter:
    @router.post("/run")
    async def run_backtest():
        return {"status": "not_implemented", "message": "Extended backtesting coming in Phase 4"}

    @router.get("/results")
    async def get_results():
        results = store.get_backtest_results()
        return {"results": results, "total": len(results)}

    @router.get("/compare")
    async def compare_results():
        return {"status": "not_implemented"}

    return router
```

`api/routes/intelligence.py`:
```python
"""Intelligence signal routes."""
from fastapi import APIRouter
from api.data_store import DataStore

router = APIRouter(prefix="/intelligence", tags=["intelligence"])

def create_router(store: DataStore) -> APIRouter:
    @router.get("/signals")
    async def get_signals():
        signals = store.get_intelligence()
        if not signals:
            return {"status": "not_enabled", "message": "Enable intelligence in .env"}
        return signals

    return router
```

`api/routes/arbitrage.py`:
```python
"""Arbitrage monitoring routes."""
from fastapi import APIRouter
from api.data_store import DataStore

router = APIRouter(prefix="/arbitrage", tags=["arbitrage"])

def create_router(store: DataStore) -> APIRouter:
    @router.get("/opportunities")
    async def get_opportunities():
        arb = store.get_arbitrage()
        if not arb:
            return {"status": "not_enabled", "message": "Enable arbitrage in .env"}
        return arb

    return router
```

`api/routes/risk.py`:
```python
"""Risk simulation routes — Monte Carlo, VaR, stress testing."""
from fastapi import APIRouter
from api.data_store import DataStore

router = APIRouter(prefix="/risk", tags=["risk"])

def create_router(store: DataStore) -> APIRouter:
    @router.get("/simulation")
    async def get_simulation():
        mc = store.get_monte_carlo()
        if not mc:
            return {"status": "not_run", "message": "Run Monte Carlo simulation first"}
        return mc

    @router.post("/monte-carlo")
    async def run_monte_carlo():
        return {"status": "not_implemented", "message": "Monte Carlo coming in Phase 7"}

    return router
```

**Step 2: Register stub routers in server.py**

In `create_app()`, after the existing router registrations, add:

```python
    from api.routes import backtest, intelligence, arbitrage, risk
    app.include_router(backtest.create_router(data_store))
    app.include_router(intelligence.create_router(data_store))
    app.include_router(arbitrage.create_router(data_store))
    app.include_router(risk.create_router(data_store))
```

**Step 3: Verify all endpoints respond**

```bash
python -m pytest tests/test_api.py -v
```
Expected: All existing tests still pass

**Step 4: Commit**

```bash
git add api/routes/backtest.py api/routes/intelligence.py api/routes/arbitrage.py api/routes/risk.py api/server.py
git commit -m "feat: add stub routes for backtest, intelligence, arbitrage, and risk"
```

---

## Phase 2: Docker Deployment

### Task 2.1: Create Dockerfile

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/Dockerfile`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/.dockerignore`

**Step 1: Create Dockerfile**

```dockerfile
# Crypto Trading Agent v6.0 — Production Container
# Runs: FastAPI server (main) + Trading Agent (background thread)

FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies (for numpy, scipy compilation)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies first (cache layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create data directory for state persistence
RUN mkdir -p /app/data

# Environment defaults
ENV DATA_DIR=/app/data
ENV API_HOST=0.0.0.0
ENV API_PORT=8000
ENV TRADING_MODE=paper
ENV PYTHONUNBUFFERED=1

# Expose API port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); r.raise_for_status()"

# Run the API server (which starts the agent in a background thread)
CMD ["python", "-m", "api.server"]
```

**Step 2: Create .dockerignore**

```
__pycache__
*.pyc
*.pyo
.env
data/
*.log
*.pkl
agent_state*.json
.git
.gitignore
node_modules
dashboard/node_modules
dashboard/.next
docs/
tests/
*.md
.vscode
.idea
```

**Step 3: Commit**

```bash
git add Dockerfile .dockerignore
git commit -m "feat: add Dockerfile for containerized agent+API deployment"
```

---

### Task 2.2: Create Docker Compose

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/docker-compose.yml`

**Step 1: Create docker-compose.yml**

```yaml
# Crypto Trading Agent v6.0 — Docker Compose
# Runs: Agent+API in one container, Dashboard in another

version: "3.8"

services:
  agent:
    build: .
    container_name: crypto-agent
    env_file: .env
    environment:
      - DATA_DIR=/app/data
      - API_HOST=0.0.0.0
      - API_PORT=8000
    ports:
      - "${API_PORT:-8000}:8000"
    volumes:
      - ./data:/app/data
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "python", "-c", "import requests; r=requests.get('http://localhost:8000/health'); r.raise_for_status()"]
      interval: 30s
      timeout: 10s
      retries: 3
      start_period: 30s

  dashboard:
    build: ./dashboard
    container_name: crypto-dashboard
    ports:
      - "3000:3000"
    environment:
      - NEXT_PUBLIC_API_URL=http://agent:8000
    depends_on:
      agent:
        condition: service_healthy
    restart: unless-stopped
```

**Step 2: Create data directory**

```bash
mkdir -p /Users/adityaraj0421/Finances/crypto-trading-agent/data
echo "# State files persisted here" > /Users/adityaraj0421/Finances/crypto-trading-agent/data/.gitkeep
```

**Step 3: Verify Docker build**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
docker build -t crypto-agent .
```
Expected: Build succeeds

**Step 4: Commit**

```bash
git add docker-compose.yml data/.gitkeep
git commit -m "feat: add Docker Compose with agent and dashboard services"
```

---

## Phase 3: React Dashboard

### Task 3.1: Scaffold Next.js Dashboard

**Files:**
- Create entire `dashboard/` directory structure

**Step 1: Initialize Next.js project**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
npx create-next-app@latest dashboard \
  --typescript \
  --tailwind \
  --app \
  --src-dir=false \
  --import-alias="@/*" \
  --no-eslint
```

**Step 2: Install dashboard dependencies**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent/dashboard
npm install recharts swr
```

**Step 3: Commit**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
git add dashboard/
git commit -m "feat: scaffold Next.js dashboard with Tailwind and dependencies"
```

---

### Task 3.2: Create API Client and Types

**Files:**
- Create: `dashboard/lib/api.ts`
- Create: `dashboard/lib/types.ts`

**Step 1: Create types**

```typescript
// dashboard/lib/types.ts
export interface Snapshot {
  cycle: number;
  price: number;
  pair: string;
  trading_pair: string;
  capital: number;
  total_pnl: number;
  daily_pnl: number;
  total_fees: number;
  win_rate: number;
  total_trades: number;
  open_positions: number;
  regime: string;
  positions: Position[];
  autonomous: AutonomousStatus;
  updated_at: string;
}

export interface Position {
  symbol: string;
  side: string;
  entry_price: number;
  quantity: number;
  unrealized_pnl: number;
  stop_loss: number;
  take_profit: number;
  trailing_stop: number;
  strategy: string;
}

export interface Trade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  quantity: number;
  pnl_gross: number;
  pnl_net: number;
  fees_paid: number;
  entry_time: string;
  exit_time: string;
  exit_reason: string;
  strategy: string;
  hold_bars: number;
}

export interface EquityPoint {
  equity: number;
  timestamp: string;
}

export interface AutonomousStatus {
  state: string;
  daily_pnl: number;
  consecutive_losses: number;
  total_autonomous_decisions: number;
  healer: Record<string, unknown>;
  evolver: Record<string, unknown>;
  meta_learner: Record<string, unknown>;
  optimizer: Record<string, unknown>;
  recent_events: AutonomousEvent[];
}

export interface AutonomousEvent {
  type: string;
  description: string;
  timestamp: string;
  data: Record<string, unknown>;
}

export interface HealthResponse {
  status: string;
  agent_running: boolean;
  last_update: string | null;
}
```

**Step 2: Create API client**

```typescript
// dashboard/lib/api.ts
const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

async function fetchAPI<T>(path: string): Promise<T> {
  const res = await fetch(`${API_URL}${path}`);
  if (!res.ok) throw new Error(`API error: ${res.status}`);
  return res.json();
}

export const api = {
  getHealth: () => fetchAPI<{ status: string; agent_running: boolean }>("/health"),
  getStatus: () => fetchAPI<Record<string, unknown>>("/status"),
  getConfig: () => fetchAPI<Record<string, unknown>>("/config"),
  getTrades: (limit = 100) => fetchAPI<{ trades: unknown[]; total: number }>(`/trades?limit=${limit}`),
  getEquity: (limit = 0) => fetchAPI<{ equity: unknown[]; total_points: number }>(`/equity?limit=${limit}`),
  getPositions: () => fetchAPI<{ positions: unknown[]; count: number }>("/positions"),
  getAutonomousStatus: () => fetchAPI<Record<string, unknown>>("/autonomous/status"),
  getAutonomousEvents: (limit = 50) => fetchAPI<{ events: unknown[]; count: number }>(`/autonomous/events?limit=${limit}`),
  getIntelligence: () => fetchAPI<Record<string, unknown>>("/intelligence/signals"),
  getArbitrage: () => fetchAPI<Record<string, unknown>>("/arbitrage/opportunities"),
  getRiskSimulation: () => fetchAPI<Record<string, unknown>>("/risk/simulation"),
  getBacktestResults: () => fetchAPI<Record<string, unknown>>("/backtest/results"),
};

export default api;
```

**Step 3: Commit**

```bash
git add dashboard/lib/
git commit -m "feat: add TypeScript types and API client for dashboard"
```

---

### Task 3.3: Create Reusable Components

**Files:**
- Create: `dashboard/components/MetricCard.tsx`
- Create: `dashboard/components/EquityChart.tsx`
- Create: `dashboard/components/TradeTable.tsx`
- Create: `dashboard/components/StatusBadge.tsx`
- Create: `dashboard/components/Sidebar.tsx`

**Step 1: Create MetricCard.tsx**

```tsx
// dashboard/components/MetricCard.tsx
interface MetricCardProps {
  label: string;
  value: string;
  subtext?: string;
  color?: "green" | "red" | "blue" | "yellow" | "default";
}

const colorMap = {
  green: "text-green-400",
  red: "text-red-400",
  blue: "text-blue-400",
  yellow: "text-yellow-400",
  default: "text-white",
};

export default function MetricCard({ label, value, subtext, color = "default" }: MetricCardProps) {
  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-4">
      <p className="text-xs text-gray-400 uppercase tracking-wide">{label}</p>
      <p className={`text-2xl font-bold mt-1 ${colorMap[color]}`}>{value}</p>
      {subtext && <p className="text-xs text-gray-500 mt-1">{subtext}</p>}
    </div>
  );
}
```

**Step 2: Create StatusBadge.tsx**

```tsx
// dashboard/components/StatusBadge.tsx
interface StatusBadgeProps {
  state: string;
}

const stateStyles: Record<string, string> = {
  normal: "bg-green-900 text-green-300 border-green-700",
  cautious: "bg-yellow-900 text-yellow-300 border-yellow-700",
  defensive: "bg-orange-900 text-orange-300 border-orange-700",
  halted: "bg-red-900 text-red-300 border-red-700",
};

export default function StatusBadge({ state }: StatusBadgeProps) {
  const style = stateStyles[state] || "bg-gray-700 text-gray-300 border-gray-600";
  return (
    <span className={`px-2 py-1 rounded-full text-xs font-medium border ${style}`}>
      {state.toUpperCase()}
    </span>
  );
}
```

**Step 3: Create EquityChart.tsx**

```tsx
// dashboard/components/EquityChart.tsx
"use client";

import { LineChart, Line, XAxis, YAxis, Tooltip, ResponsiveContainer, Area, AreaChart } from "recharts";

interface EquityChartProps {
  data: { equity: number; timestamp: string }[];
  height?: number;
}

export default function EquityChart({ data, height = 300 }: EquityChartProps) {
  if (data.length === 0) {
    return (
      <div className="flex items-center justify-center h-48 bg-gray-800 rounded-lg border border-gray-700">
        <p className="text-gray-500">No equity data yet. Waiting for agent cycles...</p>
      </div>
    );
  }

  const formatted = data.map((d) => ({
    ...d,
    time: new Date(d.timestamp).toLocaleTimeString(),
  }));

  return (
    <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
      <ResponsiveContainer width="100%" height={height}>
        <AreaChart data={formatted}>
          <defs>
            <linearGradient id="equityGrad" x1="0" y1="0" x2="0" y2="1">
              <stop offset="5%" stopColor="#3b82f6" stopOpacity={0.3} />
              <stop offset="95%" stopColor="#3b82f6" stopOpacity={0} />
            </linearGradient>
          </defs>
          <XAxis dataKey="time" stroke="#6b7280" tick={{ fontSize: 10 }} />
          <YAxis stroke="#6b7280" tick={{ fontSize: 10 }} />
          <Tooltip
            contentStyle={{ backgroundColor: "#1f2937", border: "1px solid #374151" }}
            labelStyle={{ color: "#9ca3af" }}
          />
          <Area
            type="monotone"
            dataKey="equity"
            stroke="#3b82f6"
            fill="url(#equityGrad)"
            strokeWidth={2}
          />
        </AreaChart>
      </ResponsiveContainer>
    </div>
  );
}
```

**Step 4: Create TradeTable.tsx**

```tsx
// dashboard/components/TradeTable.tsx
interface Trade {
  symbol: string;
  side: string;
  entry_price: number;
  exit_price: number;
  pnl_net: number;
  fees_paid: number;
  exit_reason: string;
  strategy: string;
  hold_bars: number;
  exit_time: string;
}

interface TradeTableProps {
  trades: Trade[];
}

export default function TradeTable({ trades }: TradeTableProps) {
  if (trades.length === 0) {
    return (
      <div className="text-center py-8 text-gray-500">
        No trades yet. The agent will execute when signals are strong enough.
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="text-gray-400 border-b border-gray-700">
            <th className="text-left py-2 px-3">Time</th>
            <th className="text-left py-2 px-3">Side</th>
            <th className="text-right py-2 px-3">Entry</th>
            <th className="text-right py-2 px-3">Exit</th>
            <th className="text-right py-2 px-3">PnL</th>
            <th className="text-left py-2 px-3">Strategy</th>
            <th className="text-left py-2 px-3">Exit Reason</th>
            <th className="text-right py-2 px-3">Bars</th>
          </tr>
        </thead>
        <tbody>
          {trades.slice().reverse().map((t, i) => (
            <tr key={i} className="border-b border-gray-800 hover:bg-gray-800/50">
              <td className="py-2 px-3 text-gray-400">
                {new Date(t.exit_time).toLocaleString()}
              </td>
              <td className={`py-2 px-3 font-medium ${t.side === "long" ? "text-green-400" : "text-red-400"}`}>
                {t.side.toUpperCase()}
              </td>
              <td className="py-2 px-3 text-right">${t.entry_price.toLocaleString()}</td>
              <td className="py-2 px-3 text-right">${t.exit_price.toLocaleString()}</td>
              <td className={`py-2 px-3 text-right font-medium ${t.pnl_net >= 0 ? "text-green-400" : "text-red-400"}`}>
                ${t.pnl_net.toFixed(2)}
              </td>
              <td className="py-2 px-3 text-gray-300">{t.strategy}</td>
              <td className="py-2 px-3 text-gray-400">{t.exit_reason}</td>
              <td className="py-2 px-3 text-right text-gray-400">{t.hold_bars}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
```

**Step 5: Create Sidebar.tsx**

```tsx
// dashboard/components/Sidebar.tsx
"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const navItems = [
  { href: "/", label: "Overview", icon: "📊" },
  { href: "/equity", label: "Equity Curve", icon: "📈" },
  { href: "/trades", label: "Trade History", icon: "💹" },
  { href: "/autonomous", label: "Autonomous", icon: "🤖" },
  { href: "/backtest", label: "Backtesting", icon: "🔬" },
  { href: "/intelligence", label: "Intelligence", icon: "🧠" },
  { href: "/arbitrage", label: "Arbitrage", icon: "⚡" },
  { href: "/risk", label: "Risk Sim", icon: "🎲" },
];

export default function Sidebar() {
  const pathname = usePathname();

  return (
    <aside className="w-56 bg-gray-900 border-r border-gray-800 min-h-screen p-4">
      <h1 className="text-lg font-bold text-blue-400 mb-6">Crypto Agent v6</h1>
      <nav className="space-y-1">
        {navItems.map((item) => {
          const active = pathname === item.href;
          return (
            <Link
              key={item.href}
              href={item.href}
              className={`flex items-center gap-2 px-3 py-2 rounded-lg text-sm transition-colors
                ${active
                  ? "bg-blue-900/50 text-blue-300 border border-blue-800"
                  : "text-gray-400 hover:text-white hover:bg-gray-800"
                }`}
            >
              <span>{item.icon}</span>
              <span>{item.label}</span>
            </Link>
          );
        })}
      </nav>
    </aside>
  );
}
```

**Step 6: Commit**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
git add dashboard/components/
git commit -m "feat: add dashboard UI components (MetricCard, EquityChart, TradeTable, StatusBadge, Sidebar)"
```

---

### Task 3.4: Create Dashboard Pages

**Files:**
- Modify: `dashboard/app/layout.tsx`
- Modify: `dashboard/app/page.tsx`
- Create: `dashboard/app/equity/page.tsx`
- Create: `dashboard/app/trades/page.tsx`
- Create: `dashboard/app/autonomous/page.tsx`
- Create: `dashboard/app/backtest/page.tsx`
- Create: `dashboard/app/intelligence/page.tsx`
- Create: `dashboard/app/arbitrage/page.tsx`
- Create: `dashboard/app/risk/page.tsx`

**Step 1: Update layout.tsx**

Replace the contents of `dashboard/app/layout.tsx`:

```tsx
import type { Metadata } from "next";
import "./globals.css";
import Sidebar from "@/components/Sidebar";

export const metadata: Metadata = {
  title: "Crypto Trading Agent v6.0",
  description: "Real-time dashboard for the autonomous trading agent",
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" className="dark">
      <body className="bg-gray-950 text-gray-100 antialiased">
        <div className="flex">
          <Sidebar />
          <main className="flex-1 p-6 overflow-auto min-h-screen">
            {children}
          </main>
        </div>
      </body>
    </html>
  );
}
```

**Step 2: Create home page (dashboard/app/page.tsx)**

```tsx
"use client";

import useSWR from "swr";
import MetricCard from "@/components/MetricCard";
import EquityChart from "@/components/EquityChart";
import StatusBadge from "@/components/StatusBadge";
import TradeTable from "@/components/TradeTable";
import api from "@/lib/api";

export default function Home() {
  const { data: status } = useSWR("/status", () => api.getStatus(), { refreshInterval: 5000 });
  const { data: equity } = useSWR("/equity", () => api.getEquity(100), { refreshInterval: 5000 });
  const { data: trades } = useSWR("/trades", () => api.getTrades(10), { refreshInterval: 5000 });

  const s = status as Record<string, any> || {};
  const isWaiting = s.status === "waiting";

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold">Dashboard</h1>
        {s.autonomous && <StatusBadge state={s.autonomous.state} />}
      </div>

      {isWaiting ? (
        <div className="text-center py-20 text-gray-500">
          <p className="text-lg">Waiting for agent to start...</p>
          <p className="text-sm mt-2">The agent needs to complete at least one cycle.</p>
        </div>
      ) : (
        <>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
            <MetricCard
              label="Capital"
              value={`$${(s.capital || 0).toLocaleString()}`}
              color="blue"
            />
            <MetricCard
              label="Total PnL"
              value={`$${(s.total_pnl || 0).toFixed(2)}`}
              color={(s.total_pnl || 0) >= 0 ? "green" : "red"}
            />
            <MetricCard
              label="Win Rate"
              value={`${((s.win_rate || 0) * 100).toFixed(1)}%`}
              subtext={`${s.total_trades || 0} trades`}
            />
            <MetricCard
              label="Regime"
              value={s.regime || "Unknown"}
              subtext={`Cycle #${s.cycle || 0}`}
              color="yellow"
            />
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-3">Equity Curve</h2>
            <EquityChart data={equity?.equity || []} height={250} />
          </div>

          <div>
            <h2 className="text-lg font-semibold mb-3">Recent Trades</h2>
            <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
              <TradeTable trades={(trades?.trades || []) as any[]} />
            </div>
          </div>
        </>
      )}
    </div>
  );
}
```

**Step 3: Create equity page**

```tsx
// dashboard/app/equity/page.tsx
"use client";
import useSWR from "swr";
import EquityChart from "@/components/EquityChart";
import api from "@/lib/api";

export default function EquityPage() {
  const { data } = useSWR("/equity", () => api.getEquity(), { refreshInterval: 5000 });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Equity Curve</h1>
      <EquityChart data={data?.equity || []} height={500} />
      <p className="text-sm text-gray-500">
        Total data points: {data?.total_points || 0}
      </p>
    </div>
  );
}
```

**Step 4: Create trades page**

```tsx
// dashboard/app/trades/page.tsx
"use client";
import useSWR from "swr";
import TradeTable from "@/components/TradeTable";
import api from "@/lib/api";

export default function TradesPage() {
  const { data } = useSWR("/trades", () => api.getTrades(500), { refreshInterval: 5000 });

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Trade History</h1>
      <p className="text-sm text-gray-400">Total trades: {data?.total || 0}</p>
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-4">
        <TradeTable trades={(data?.trades || []) as any[]} />
      </div>
    </div>
  );
}
```

**Step 5: Create autonomous page**

```tsx
// dashboard/app/autonomous/page.tsx
"use client";
import useSWR from "swr";
import StatusBadge from "@/components/StatusBadge";
import MetricCard from "@/components/MetricCard";
import api from "@/lib/api";

export default function AutonomousPage() {
  const { data: status } = useSWR("/autonomous/status", () => api.getAutonomousStatus(), { refreshInterval: 5000 });
  const { data: events } = useSWR("/autonomous/events", () => api.getAutonomousEvents(), { refreshInterval: 5000 });

  const s = status as Record<string, any> || {};

  return (
    <div className="space-y-6">
      <div className="flex items-center gap-3">
        <h1 className="text-2xl font-bold">Autonomous System</h1>
        <StatusBadge state={s.state || "unknown"} />
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-4">
        <MetricCard label="Daily PnL" value={`$${(s.daily_pnl || 0).toFixed(2)}`} color={(s.daily_pnl || 0) >= 0 ? "green" : "red"} />
        <MetricCard label="Loss Streak" value={String(s.consecutive_losses || 0)} color={(s.consecutive_losses || 0) >= 3 ? "red" : "default"} />
        <MetricCard label="Decisions" value={String(s.total_autonomous_decisions || 0)} />
        <MetricCard label="State" value={s.state || "unknown"} color="yellow" />
      </div>

      <div>
        <h2 className="text-lg font-semibold mb-3">Recent Events</h2>
        <div className="bg-gray-800 rounded-lg border border-gray-700 p-4 space-y-2">
          {(events?.events || []).length === 0 ? (
            <p className="text-gray-500">No events yet.</p>
          ) : (
            (events?.events as any[] || []).slice().reverse().map((e: any, i: number) => (
              <div key={i} className="flex items-center gap-3 py-1 border-b border-gray-700/50 last:border-0">
                <span className="text-xs text-gray-500">{new Date(e.timestamp).toLocaleString()}</span>
                <span className="text-xs font-medium text-blue-400">[{e.type}]</span>
                <span className="text-sm text-gray-300">{e.description}</span>
              </div>
            ))
          )}
        </div>
      </div>
    </div>
  );
}
```

**Step 6: Create placeholder pages for phases 4-7**

Each of these follows the same pattern — "Coming Soon" with a brief description:

```tsx
// dashboard/app/backtest/page.tsx
export default function BacktestPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Backtesting</h1>
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <p className="text-4xl mb-4">🔬</p>
        <p className="text-lg text-gray-300">Extended Backtesting</p>
        <p className="text-sm text-gray-500 mt-2">Multi-pair, multi-scenario, multi-timeframe stress testing. Coming in Phase 4.</p>
      </div>
    </div>
  );
}
```

```tsx
// dashboard/app/intelligence/page.tsx
export default function IntelligencePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Intelligence Signals</h1>
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <p className="text-4xl mb-4">🧠</p>
        <p className="text-lg text-gray-300">On-Chain + NLP + Whale Tracking</p>
        <p className="text-sm text-gray-500 mt-2">On-chain analytics, order book depth, stock correlation, whale tracking, news sentiment. Coming in Phase 5.</p>
      </div>
    </div>
  );
}
```

```tsx
// dashboard/app/arbitrage/page.tsx
export default function ArbitragePage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Multi-Exchange Arbitrage</h1>
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <p className="text-4xl mb-4">⚡</p>
        <p className="text-lg text-gray-300">Binance × Coinbase × Kraken</p>
        <p className="text-sm text-gray-500 mt-2">Cross-exchange price monitoring, fee-adjusted opportunity detection, simultaneous execution. Coming in Phase 6.</p>
      </div>
    </div>
  );
}
```

```tsx
// dashboard/app/risk/page.tsx
export default function RiskPage() {
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-bold">Risk Simulation</h1>
      <div className="bg-gray-800 rounded-lg border border-gray-700 p-8 text-center">
        <p className="text-4xl mb-4">🎲</p>
        <p className="text-lg text-gray-300">Monte Carlo + Black Swan Stress Tests</p>
        <p className="text-sm text-gray-500 mt-2">10,000 simulation paths, VaR/CVaR, probability of ruin, historical crash replays. Coming in Phase 7.</p>
      </div>
    </div>
  );
}
```

**Step 7: Verify dashboard builds**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent/dashboard
npm run build
```
Expected: Build succeeds

**Step 8: Commit**

```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
git add dashboard/
git commit -m "feat: add all dashboard pages with live data, equity chart, trade table, and placeholder pages"
```

---

## Phase 4: Extended Backtesting

### Task 4.1: Create Market Scenario Generator

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/scenarios.py`

**Step 1: Create scenarios.py**

```python
"""
Market Scenario Generator
===========================
Generates synthetic OHLCV data for specific market conditions.
Used for stress-testing strategies before going live.
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_scenario(scenario: str, periods: int = 500,
                      base_price: float = 100000.0,
                      timeframe_minutes: int = 60) -> pd.DataFrame:
    """
    Generate synthetic OHLCV data for a named scenario.

    Scenarios:
        bull_run, bear_market, sideways_chop, flash_crash,
        black_swan, accumulation
    """
    generators = {
        "bull_run": _bull_run,
        "bear_market": _bear_market,
        "sideways_chop": _sideways_chop,
        "flash_crash": _flash_crash,
        "black_swan": _black_swan,
        "accumulation": _accumulation,
    }
    if scenario not in generators:
        raise ValueError(f"Unknown scenario: {scenario}. Available: {list(generators.keys())}")

    prices = generators[scenario](periods, base_price)
    return _prices_to_ohlcv(prices, timeframe_minutes)


def _bull_run(periods: int, base: float) -> np.ndarray:
    """Steady uptrend with pullbacks. +50-100% over period."""
    drift = 0.0008  # ~0.08% per bar upward
    volatility = 0.015
    returns = np.random.normal(drift, volatility, periods)
    # Add 2-3 pullbacks of 5-10%
    for _ in range(3):
        start = np.random.randint(periods // 5, periods - 20)
        returns[start:start + 10] = np.random.normal(-0.008, 0.02, 10)
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _bear_market(periods: int, base: float) -> np.ndarray:
    """Sustained downtrend with relief rallies. -40-60% over period."""
    drift = -0.0006
    volatility = 0.018
    returns = np.random.normal(drift, volatility, periods)
    # Relief rallies
    for _ in range(3):
        start = np.random.randint(periods // 5, periods - 15)
        returns[start:start + 8] = np.random.normal(0.006, 0.015, 8)
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _sideways_chop(periods: int, base: float) -> np.ndarray:
    """Range-bound with false breakouts. ±5% around base."""
    volatility = 0.012
    returns = np.random.normal(0, volatility, periods)
    # Mean-revert: pull back toward base
    prices = [base]
    for r in returns:
        new_price = prices[-1] * (1 + r)
        # Mean reversion force
        deviation = (new_price - base) / base
        new_price *= (1 - deviation * 0.05)
        prices.append(new_price)
    return np.array(prices[1:])


def _flash_crash(periods: int, base: float) -> np.ndarray:
    """Normal market, then sudden 30% crash, then V-shaped recovery."""
    prices = np.zeros(periods)
    # Normal phase (60% of data)
    normal_end = int(periods * 0.6)
    drift = 0.0003
    vol = 0.012
    returns = np.random.normal(drift, vol, normal_end)
    prices[:normal_end] = base * np.exp(np.cumsum(returns))

    # Crash phase (5% of data — sharp drop)
    crash_bars = max(int(periods * 0.05), 5)
    crash_returns = np.random.normal(-0.06, 0.03, crash_bars)
    crash_start = prices[normal_end - 1]
    crash_prices = crash_start * np.exp(np.cumsum(crash_returns))
    prices[normal_end:normal_end + crash_bars] = crash_prices

    # Recovery phase (35% of data)
    recovery_start = normal_end + crash_bars
    recovery_bars = periods - recovery_start
    if recovery_bars > 0:
        recovery_returns = np.random.normal(0.004, 0.02, recovery_bars)
        recovery_base = prices[recovery_start - 1]
        prices[recovery_start:] = recovery_base * np.exp(np.cumsum(recovery_returns))

    return prices


def _black_swan(periods: int, base: float) -> np.ndarray:
    """COVID-style: 50% crash in 2 days, slow multi-month recovery."""
    prices = np.zeros(periods)
    # Pre-crash (40%)
    pre = int(periods * 0.4)
    returns = np.random.normal(0.0004, 0.01, pre)
    prices[:pre] = base * np.exp(np.cumsum(returns))

    # Crash (2% of data — extremely fast)
    crash_bars = max(int(periods * 0.02), 3)
    crash_returns = np.random.normal(-0.15, 0.05, crash_bars)
    crash_base = prices[pre - 1]
    prices[pre:pre + crash_bars] = crash_base * np.exp(np.cumsum(crash_returns))

    # Slow recovery (58%)
    recovery_start = pre + crash_bars
    recovery_bars = periods - recovery_start
    if recovery_bars > 0:
        recovery_returns = np.random.normal(0.002, 0.015, recovery_bars)
        rec_base = prices[recovery_start - 1]
        prices[recovery_start:] = rec_base * np.exp(np.cumsum(recovery_returns))

    return prices


def _accumulation(periods: int, base: float) -> np.ndarray:
    """Low volatility consolidation. Tight range, decreasing volume."""
    volatility = 0.005  # Very low
    returns = np.random.normal(0, volatility, periods)
    # Tighten volatility over time
    decay = np.linspace(1.0, 0.3, periods)
    returns *= decay
    prices = base * np.exp(np.cumsum(returns))
    return prices


def _prices_to_ohlcv(prices: np.ndarray, tf_minutes: int) -> pd.DataFrame:
    """Convert close prices to OHLCV DataFrame with realistic open/high/low/volume."""
    n = len(prices)
    now = datetime.now()
    timestamps = [now - timedelta(minutes=tf_minutes * (n - i)) for i in range(n)]

    volatility = np.std(np.diff(prices) / prices[:-1]) if n > 1 else 0.01
    spread = prices * volatility * 0.5

    opens = prices * (1 + np.random.uniform(-0.002, 0.002, n))
    highs = np.maximum(prices, opens) + np.abs(np.random.normal(0, 1, n)) * spread
    lows = np.minimum(prices, opens) - np.abs(np.random.normal(0, 1, n)) * spread
    volumes = np.random.lognormal(10, 1.5, n) * (prices / prices[0])

    df = pd.DataFrame({
        "open": opens,
        "high": highs,
        "low": lows,
        "close": prices,
        "volume": volumes,
    }, index=pd.DatetimeIndex(timestamps, name="timestamp"))

    return df


def list_scenarios() -> list[str]:
    """Return all available scenario names."""
    return ["bull_run", "bear_market", "sideways_chop", "flash_crash", "black_swan", "accumulation"]
```

**Step 2: Commit**

```bash
git add scenarios.py
git commit -m "feat: add market scenario generator for stress-testing"
```

---

### Task 4.2: Extend Backtester for Multi-Pair and Multi-Scenario

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/backtest_runner.py`

**Step 1: Create backtest_runner.py**

```python
"""
Backtest Runner v2.0
=====================
Orchestrates multi-pair, multi-scenario, multi-timeframe backtests.
Results are stored in DataStore for dashboard display.
"""

from datetime import datetime
from backtester import Backtester
from scenarios import generate_scenario, list_scenarios
from data_fetcher import DataFetcher
from config import Config


class BacktestRunner:
    """Orchestrates complex backtest runs."""

    def __init__(self):
        self.results: list[dict] = []
        self.fetcher = DataFetcher()

    def run_scenario(self, scenario: str, periods: int = 500,
                     base_price: float = 100000.0) -> dict:
        """Run backtest on a synthetic market scenario."""
        df = generate_scenario(scenario, periods=periods, base_price=base_price)
        bt = Backtester(
            fee_pct=Config.FEE_PCT,
            slippage_pct=Config.SLIPPAGE_PCT,
        )
        metrics = bt.run(df, verbose=False)
        result = {
            "type": "scenario",
            "scenario": scenario,
            "pair": "SYNTHETIC",
            "timeframe": Config.TIMEFRAME,
            "periods": periods,
            "run_at": datetime.now().isoformat(),
            "metrics": metrics,
        }
        self.results.append(result)
        return result

    def run_multi_pair(self, pairs: list[str] = None,
                       limit: int = 500) -> list[dict]:
        """Run backtests across multiple trading pairs."""
        pairs = pairs or Config.TRADING_PAIRS
        results = []
        for pair in pairs:
            try:
                df = self.fetcher.fetch_ohlcv(symbol=pair, limit=limit)
                if df.empty or len(df) < 100:
                    results.append({
                        "type": "multi_pair",
                        "pair": pair,
                        "error": "insufficient_data",
                    })
                    continue

                bt = Backtester(
                    fee_pct=Config.FEE_PCT,
                    slippage_pct=Config.SLIPPAGE_PCT,
                )
                metrics = bt.run(df, verbose=False)
                result = {
                    "type": "multi_pair",
                    "pair": pair,
                    "timeframe": Config.TIMEFRAME,
                    "periods": len(df),
                    "run_at": datetime.now().isoformat(),
                    "metrics": metrics,
                }
                results.append(result)
                self.results.append(result)
            except Exception as e:
                results.append({"type": "multi_pair", "pair": pair, "error": str(e)})

        return results

    def run_all_scenarios(self) -> list[dict]:
        """Run backtests on all predefined market scenarios."""
        results = []
        for scenario in list_scenarios():
            result = self.run_scenario(scenario)
            results.append(result)
        return results

    def run_multi_timeframe(self, timeframes: list[str] = None,
                            limit: int = 500) -> list[dict]:
        """Run backtests across multiple timeframes."""
        timeframes = timeframes or ["15m", "1h", "4h"]
        results = []
        for tf in timeframes:
            try:
                df = self.fetcher.fetch_ohlcv(timeframe=tf, limit=limit)
                if df.empty or len(df) < 100:
                    continue
                bt = Backtester()
                metrics = bt.run(df, verbose=False)
                result = {
                    "type": "multi_timeframe",
                    "pair": Config.TRADING_PAIR,
                    "timeframe": tf,
                    "periods": len(df),
                    "run_at": datetime.now().isoformat(),
                    "metrics": metrics,
                }
                results.append(result)
                self.results.append(result)
            except Exception as e:
                results.append({"type": "multi_timeframe", "timeframe": tf, "error": str(e)})

        return results

    def get_all_results(self) -> list[dict]:
        return self.results
```

**Step 2: Wire into API backtest route**

Update `api/routes/backtest.py`:

```python
"""Backtest routes — run and view backtest results."""
import threading
from fastapi import APIRouter, Query
from pydantic import BaseModel
from typing import Optional
from api.data_store import DataStore

router = APIRouter(prefix="/backtest", tags=["backtest"])


class BacktestRequest(BaseModel):
    pair: Optional[str] = None
    scenario: Optional[str] = None
    timeframe: Optional[str] = None
    periods: int = 500


def create_router(store: DataStore) -> APIRouter:

    @router.post("/run")
    async def run_backtest(req: BacktestRequest):
        """Run a backtest in a background thread."""
        def _run():
            from backtest_runner import BacktestRunner
            runner = BacktestRunner()
            if req.scenario:
                result = runner.run_scenario(req.scenario, periods=req.periods)
            elif req.pair:
                result = runner.run_multi_pair([req.pair])[0] if runner.run_multi_pair([req.pair]) else {}
            else:
                result = runner.run_all_scenarios()
            # Store results
            existing = store.get_backtest_results()
            if isinstance(result, list):
                existing.extend(result)
            else:
                existing.append(result)
            store.update_backtest_results(existing)

        thread = threading.Thread(target=_run, daemon=True)
        thread.start()
        return {"status": "started", "message": "Backtest running in background"}

    @router.get("/results")
    async def get_results():
        results = store.get_backtest_results()
        return {"results": results, "total": len(results)}

    @router.get("/scenarios")
    async def list_scenarios():
        from scenarios import list_scenarios as ls
        return {"scenarios": ls()}

    return router
```

**Step 3: Commit**

```bash
git add backtest_runner.py api/routes/backtest.py scenarios.py
git commit -m "feat: add multi-pair, multi-scenario, multi-timeframe backtest runner with API"
```

---

## Phase 5: Intelligence Signals

### Task 5.1: Create Intelligence Module

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/__init__.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/onchain.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/orderbook.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/correlation.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/whale_tracker.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/news_sentiment.py`
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/intelligence/aggregator.py`

Each intelligence provider follows the same interface pattern and returns an `IntelligenceSignal`. See design doc for full implementation details. The aggregator combines them into a single `adjustment_factor`.

**Commit after each file**, then wire the aggregator into `agent.py` at the end.

---

## Phase 6: Multi-Exchange Arbitrage

### Task 6.1: Create Arbitrage Module

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/arbitrage/__init__.py`
- Create: `arbitrage/price_monitor.py`
- Create: `arbitrage/opportunity_detector.py`
- Create: `arbitrage/fee_calculator.py`
- Create: `arbitrage/execution_engine.py`
- Create: `arbitrage/latency_tracker.py`

The arbitrage module uses async CCXT calls to fetch prices from Binance, Coinbase, and Kraken concurrently. See design doc for full implementation details.

---

## Phase 7: Monte Carlo Risk Simulation

### Task 7.1: Create Risk Simulation Module

**Files:**
- Create: `/Users/adityaraj0421/Finances/crypto-trading-agent/risk_simulation/__init__.py`
- Create: `risk_simulation/monte_carlo.py`
- Create: `risk_simulation/scenarios.py`
- Create: `risk_simulation/var_calculator.py`
- Create: `risk_simulation/visualizer.py`

The Monte Carlo simulator takes trade history, calculates return distribution, samples 10,000 paths, and computes VaR/CVaR/probability of ruin. See design doc for full implementation details.

---

## Final Integration & Verification

### Task F.1: End-to-End Verification

**Step 1: Run all Python tests**
```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
python -m pytest tests/ -v
```

**Step 2: Start the API server**
```bash
python -m api.server
```

**Step 3: Verify API endpoints**
```bash
curl http://localhost:8000/health
curl http://localhost:8000/status
curl http://localhost:8000/config
curl http://localhost:8000/trades
curl http://localhost:8000/equity
curl http://localhost:8000/autonomous/status
curl http://localhost:8000/backtest/scenarios
curl http://localhost:8000/intelligence/signals
curl http://localhost:8000/arbitrage/opportunities
curl http://localhost:8000/risk/simulation
```

**Step 4: Start the dashboard**
```bash
cd dashboard && npm run dev
# Open http://localhost:3000
```

**Step 5: Docker build and run**
```bash
cd /Users/adityaraj0421/Finances/crypto-trading-agent
docker compose build
docker compose up
```

**Step 6: Final commit**
```bash
git add -A
git commit -m "feat: crypto trading agent v6.0 — full-stack platform with dashboard, Docker, intelligence, arbitrage, and risk simulation"
```
